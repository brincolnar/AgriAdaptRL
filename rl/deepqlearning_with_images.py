import csv
import torch
import random
import os 
import torch
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
from utils_images import evaluate_model, plot_test_split
from hyperparams import EPISODES, EPSILON, EPSILON_START, EPSILON_END, EPSILON_DECAY, GAMMA, LEARNING_RATE, PERFORMANCE_FACTOR, NUM_ITERATIONS, BATCH_SIZE, TARGET_UPDATE

network_performance_columns = [
    '0%', '25%', '50%', '75%'
]

# Create a mapping from action index to network name
action_to_network_mapping = {index: name for index, name in enumerate(network_performance_columns)}

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

    features = pd.merge(features_file, performance_file, left_on='Filename', right_on='Filename')
    performance_columns = ['0%', '25%', '50%', '75%']
    features['best_network'] = features[performance_columns].idxmax(axis=1)

    train_filenames = os.listdir('./data/ordered_train_test/train/images/')
    test_filenames = os.listdir('./data/ordered_train_test/test/images')

    # Train-test split
    train_df = features[features['Filename'].isin(train_filenames)]
    test_df = features[features['Filename'].isin(test_filenames)]

    combined_df = pd.concat([train_df, test_df])

    verbose = True 

    results = []

    test_df = features[features['Filename'].isin(test_filenames)]
    
    train_env = NetworkSelectionEnvImages(dataframe=train_df, 
    image_folder='./data/ordered_train_test/train/images/', 
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

        state = train_env.reset()  # Reset the environment
        state = state.to(device).unsqueeze(0)

        for t in tqdm(range(len(train_env.dataframe)), desc=f'Iteration {i}'):
            # Select action
            if random.random() > epsilon:
                with torch.no_grad():
                    action = model(state).argmax().item()
            else:
                action = train_env.sample_action()
            
            epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** t))  # Decay epsilon

            # Execute action
            next_state, reward, done = train_env.get_next_state_reward(action)
            next_state = next_state.unsqueeze(0) 

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

                print(f'Loss at step {t}: {loss.item()}')

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

            # # Compute the target Q-value
            # next_max_q = model(next_state).max().item()  # Get the maximum Q-value for the next state
            # target_q_value = reward + GAMMA * next_max_q  # Apply the Bellman Equation
            # target_q_value = torch.tensor([target_q_value], device=device, dtype=torch.float32)

            # if verbose:
            #     print(f"Target Q-value: {target_q_value}")

            # q_values = model(state)
            # prediction = q_values[0][action].unsqueeze(0)

            # # Get the predicted Q-value for the action taken
            # current_q_values = model(state)
            # current_q_value = current_q_values.squeeze(0)[action]  

            # if verbose:
            #     print(f"Current Q-value: {current_q_value}")
                
            # # Calculate the loss
            # loss = loss_fn(current_q_value.unsqueeze(0), target_q_value)  # Unsqueeze to add the batch dimension back for loss calculation
            
            # if verbose:
            #     print(f"Loss: {loss.item()}")
            # losses.append(loss)

            # # Backpropagation
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            # state = next_state

        plt.figure(figsize=(10,5))
        plt.plot(losses, label='Loss')
        plt.xlabel('Training Steps')    
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"./loss.png", bbox_inches='tight')

        actions_taken = []

        # Evaluate the model
        total_test_reward, accuracy, average_weight, average_iou, correct_predictions, total_predictions, actions_taken = evaluate_model(
            model=model,
            optimizer=optimizer, 
            test_data=test_df, 
            performance_factor=PERFORMANCE_FACTOR,
        )
        print(f"Average accuracy: {accuracy:.2f}")
        print(f"Average IoU: {average_iou:.2f}")
        print(f"Average weight: {average_weight:.2f}")
        print(f"Correct Predictions: {correct_predictions}, Total Predictions: {total_predictions}")


    print("----------------------------------------------------------")
