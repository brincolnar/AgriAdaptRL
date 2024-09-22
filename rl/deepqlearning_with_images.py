import csv
import torch
import random
import os 
import torch
import math
import torch.nn.functional as F
import re as re
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from NetworkSelectionEnvImages import NetworkSelectionEnvImages
from ReplayBuffer import ReplayBuffer
from utils_images import evaluate_model, plot_test_split, moving_average, evaluate_test
from hyperparams import EPISODES, EPSILON, EPSILON_START, EPSILON_END, EPSILON_DECAY, GAMMA, LEARNING_RATE, PERFORMANCE_FACTOR, NUM_ITERATIONS, BATCH_SIZE, TARGET_UPDATE

network_performance_columns = [
    '0%', '25%', '50%', '75%'
]

# Create a mapping from action index to network name
action_to_network_mapping = {index: name for index, name in enumerate(network_performance_columns)}

network_to_weight_mapping = [100, 75, 50, 25]

class ConvDQN(nn.Module):
    def __init__(self, input_shape=(3, 84, 84)):  # Assuming 3 channels and 84x84 images      
        super(ConvDQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the output size of the last convolutional layer
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, 4)  # Assuming 4 actions

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

if __name__ == "__main__":

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Original network
    model = ConvDQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Target network
    target_model = ConvDQN().to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()  # Set the target network to evaluation mode

    loss_fn = nn.SmoothL1Loss() # nn.MSELoss(), nn.SmoothL1Loss() (other losses)

    # Load the dataset
    features_file = pd.read_csv('./features.csv')
    performance_file = pd.read_csv('./performance_results.csv')

    combined_df = pd.merge(features_file, performance_file, left_on='Filename', right_on='Filename')
    performance_columns = ['0%', '25%', '50%', '75%']
    combined_df['best_network'] = combined_df[performance_columns].idxmax(axis=1)
    combined_df = combined_df[~(combined_df[network_performance_columns] == 0).all(axis=1)]

    verbose = True 

    results = []
    
    env = NetworkSelectionEnvImages(dataframe=combined_df, 
    image_folder='./data/ordered_train_test/all/images/', 
    performance_factor=PERFORMANCE_FACTOR,
    device='cuda',
    verbose=False)

    replay_buffer = ReplayBuffer(capacity=BATCH_SIZE)


    # Train
    for i in range(NUM_ITERATIONS):
        epsilon = EPSILON_START
        losses = []

        print("========================================================")
        print(f"Iteration {i} starting...")
        print("========================================================")

        
        state = env.reset()  # Reset the environment
        state = state.to(device).unsqueeze(0)

        filenames = []
        ious = []
        weights = []
        battery_levels = []

        for t in tqdm(range(len(env.dataframe)), desc=f'Iteration {i}'):
            # Select action
            if random.random() > epsilon:
                with torch.no_grad():
                    action = model(state).argmax().item()
            else:
                action = env.sample_action()
            
            # epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** t))  # Decay epsilon
            epsilon = EPSILON

            # Execute action
            next_state, reward, done = env.get_next_state_reward(action)
            next_state = next_state.unsqueeze(0) 

            filenames.append(env.dataframe.iloc[env.current_idx]['Filename'])
            ious.append(env.dataframe.iloc[env.current_idx][action_to_network_mapping[action]])
            weights.append(network_to_weight_mapping[action])
            battery_levels.append(env.battery)
            
            replay_buffer.push(state, action, reward, next_state, done)

            if len(replay_buffer) >= BATCH_SIZE:
                transitions = replay_buffer.sample(BATCH_SIZE)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                batch_state = torch.cat(batch_state).float()
                batch_action = torch.tensor(batch_action, device=device).long()  # Actions are usually expected to be long
                batch_reward = torch.tensor(batch_reward, device=device).float()
                batch_next_state = torch.cat(batch_next_state).float()
                batch_done = torch.tensor(batch_done, device=device).float()

                # Compute Q-values and loss 
                current_q_values = model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
                next_q_values = target_model(batch_next_state).max(1)[0].detach()  # Detach from graph to avoid backpropagation to target model
                expected_q_values = batch_reward + GAMMA * next_q_values

                # Update target network
                if t % TARGET_UPDATE == 0:
                    target_model.load_state_dict(model.state_dict())

                loss = loss_fn(current_q_values, expected_q_values)
                losses.append(loss.item())

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
        
        results_df = pd.DataFrame({
            'Filename': filenames,
            'DQN': ious,
            'DQNWeight': weights
        })

        results_df.to_csv('deepqlearning_results.csv', index=False)
        print("Results have been saved to 'deepqlearning_results.csv'.")

        average_weight = sum(weights) / len(weights)
        average_iou = sum(ious) / len(ious) 
        last_battery_level = battery_levels[-1]
        
        print(f"Average Weight: {average_weight}")
        print(f"Average IoU: {average_iou}")
        print(f"Last Battery Level: {last_battery_level}")

        # evaluate_test(ious, weights, battery_levels)

        total_images = len(weights)
        third = total_images // 3

        first_third_avg_weight = sum(weights[:third]) / third
        second_third_avg_weight = sum(weights[third:2*third]) / third
        third_third_avg_weight = sum(weights[2*third:]) / (total_images - 2*third)

        moving_average_ious = moving_average(ious, window_size=5)
        moving_average_weights = moving_average(weights, window_size=5)
        moving_average_battery_levels = moving_average(battery_levels, window_size=5)

        plt.figure(figsize=(15, 15))

        # Plot for IoUs
        plt.subplot(3, 1, 1)  # This means 3 rows, 1 column, and this plot is the 1st
        plt.plot(moving_average_ious, label='IoU', color='blue')
        plt.xlabel('Image No.')
        plt.ylabel('IoU')
        plt.title('IoU Over Images')
        plt.legend()

        # Plot for Weights
        plt.subplot(3, 1, 2)  # This means 3 rows, 1 column, and this plot is the 2nd
        plt.plot(moving_average_weights, label='Weight', color='green')
        plt.xlabel('Image No.')
        plt.ylabel('Weight')
        plt.title('Weight Over Images')
        plt.legend()

        plt.text(x=max(len(weights)*0.1, 10), y=max(weights)*0.8, s=f'1st Third Avg Weight: {first_third_avg_weight:.2f}')
        plt.text(x=max(len(weights)*0.45, 10), y=max(weights)*0.8, s=f'2nd Third Avg Weight: {second_third_avg_weight:.2f}')
        plt.text(x=max(len(weights)*0.8, 10), y=max(weights)*0.8, s=f'3rd Third Avg Weight: {third_third_avg_weight:.2f}')

        # Plot for Battery Levels
        plt.subplot(3, 1, 3)  # This means 3 rows, 1 column, and this plot is the 3rd
        plt.plot(moving_average_battery_levels, label='Battery Level', color='red')
        plt.xlabel('Image No.')
        plt.ylabel('Battery Level')
        plt.title('Battery Level Over Images')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"./results.png", bbox_inches='tight')
