import csv
import torch
import random
import os 
import sys
import torch
import math
import torch.nn.functional as F
import re as re
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from NetworkSelectionEnvImages_ import NetworkSelectionEnvImages
from ReplayBuffer import ReplayBuffer
from utils_images import evaluate_model, plot_test_split, moving_average, evaluate_test
from hyperparams import EPISODES, EPSILON, EPSILON_START, EPSILON_END, EPSILON_DECAY, GAMMA, LEARNING_RATE, NUM_ITERATIONS, BATCH_SIZE, TARGET_UPDATE, INFERENCE_DIM
from datetime import datetime

network_performance_columns = [
    '0%', '25%', '50%', '75%'
]

# mapping from action index to network name
action_to_network_mapping = {index: name for index, name in enumerate(network_performance_columns)}

network_to_weight_mapping = [100, 75, 50, 25]

class ConvDQN(nn.Module):
    def __init__(self, input_shape=(3, 84, 84)):  
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
        self.fc2 = nn.Linear(512, 4) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def write_timing_analysis(timing_stats, performance_metrics, output_file="dqn_timing_analysis.txt"):
    """
    Write timing analysis statistics and performance metrics to a text file.
    
    Args:
        timing_stats (dict): Dictionary containing timing statistics for each operation
        performance_metrics (dict): Dictionary containing average IoU, weight, battery and process time
        output_file (str): Path to output file (default: dqn_timing_analysis.txt)
    """
    
    with open(output_file, 'w') as f:
        f.write("DQN Performance and Timing Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        from datetime import datetime
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Performance Metrics\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average IoU: {performance_metrics['average_iou']:.4f}\n")
        f.write(f"Average Weight: {performance_metrics['average_weight']:.4f}\n")
        f.write(f"Final Battery Level: {performance_metrics['last_battery_level']:.4f}\n")
        f.write(f"Average Process Time per Image: {performance_metrics['avg_image_time']:.4f} seconds\n\n")
        
        f.write("Detailed Timing Analysis\n")
        f.write("-" * 20 + "\n")
        f.write(f"{'Operation':<20} {'Mean (s)':<15} {'Std Dev (s)':<15} {'Count':<10}\n")
        f.write("-" * 60 + "\n")
        
        total_mean = sum(stats['mean'] for stats in timing_stats.values())
        total_std = (sum(stats['std_dev']**2 for stats in timing_stats.values())) ** 0.5
        # Write individual operation statistics
        for operation, stats in timing_stats.items():
            f.write(f"{operation:<20} {stats['mean']:>8.4f}s     {stats['std_dev']:>8.4f}s     {stats['count']:>5}\n")
        
        f.write("-" * 60 + "\n")
        f.write(f"{'Total':<20} {total_mean:>8.4f}s     {total_std:>8.4f}s\n\n")
        
        f.write("\nTiming Summary:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Total processing time: {total_mean:.4f} seconds\n")
        f.write(f"Average time per operation: {total_mean/len(timing_stats):.4f} seconds\n")
        operation_with_max_time = max(timing_stats.items(), key=lambda x: x[1]['mean'])[0]
        f.write(f"Most time-consuming operation: {operation_with_max_time} ({timing_stats[operation_with_max_time]['mean']:.4f}s)\n")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    performance_factor = float(sys.argv[1])

    print('=================================================')
    print(f'DEVICE: {device}')
    print(f'PERFORMANCE FACTOR: {performance_factor}')
    print(f'INFERENCE DIMENSIONS: {INFERENCE_DIM}')
    print('=================================================')
    
    # Original network
    model = ConvDQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Target network
    target_model = ConvDQN().to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()  

    loss_fn = nn.SmoothL1Loss() # nn.MSELoss(), nn.SmoothL1Loss() (other losses)

    verbose = True 
    results = []
    
    # Initialize environment with model paths
    env = NetworkSelectionEnvImages(
        image_folder='./data/ordered_train_test/all/images/',
        label_folder='./data/ordered_train_test/all/labels/', 
        performance_factor=performance_factor,
        device=device,
        model_paths=[
            "./garage/unet_512_pruned_00_iterative_1.pt",          # unpruned model
            "./garage/unet_512_pruned_025_iterative_1.pt",         # 25% pruned model
            "./garage/unet_512_pruned_05_iterative_1.pt",          # 50% pruned model
            "./garage/unet_512_pruned_075_iterative_1.pt"          # 75% pruned model
        ],
        resize_dim=(84, 84),  # For DQN input
        inference_dim=INFERENCE_DIM,
    )   


    replay_buffer = ReplayBuffer(capacity=BATCH_SIZE)

    total_start_time = time()

    # Train
    for i in range(NUM_ITERATIONS):
        iteration_start_time = time()
        epsilon = EPSILON_START
        losses = []

        print("========================================================")
        print(f"Iteration {i} starting...")
        print("========================================================")

        state = env.reset()  
        state = state.to(device).unsqueeze(0)

        filenames = []
        ious = []
        weights = []
        battery_levels = []
        image_times = []

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f'dqn_{performance_factor}_{timestamp}.csv'
        csv_path = os.path.join('./results/dqn/', csv_filename)

        with open(csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Filename', 'Weight', 'Battery Level', 'IoU'])

            num_images = len(env.image_filenames)

            for t in tqdm(range(num_images), desc=f'Iteration {i}'): 
                image_start_time = time()
                
                # Select action
                if random.random() > epsilon:
                    with torch.no_grad():
                        action = model(state).argmax().item()
                else:
                    action = env.sample_action()
                
                epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** t))  # Decay epsilon
                # epsilon = EPSILON

                # Execute action
                next_state, reward, iou, done = env.get_next_state_reward(action)
                next_state = next_state.unsqueeze(0) 

                current_filename = env.image_filenames[env.current_idx]
                filenames.append(current_filename)
                ious.append(iou)
                weight = network_to_weight_mapping[action]
                weights.append(weight)
                battery_levels.append(env.battery)

                csvwriter.writerow([current_filename, weight, env.battery, iou ])

                print(f"Current filename: {current_filename}, IoU: {iou}")

                replay_buffer.push(state, action, reward, next_state, done)

                if len(replay_buffer) >= BATCH_SIZE:
                    transitions = replay_buffer.sample(BATCH_SIZE)
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                    batch_state = torch.cat(batch_state).float()
                    batch_action = torch.tensor(batch_action, device=device).long()  
                    batch_reward = torch.tensor(batch_reward, device=device).float()
                    batch_next_state = torch.cat(batch_next_state).float()
                    batch_done = torch.tensor(batch_done, device=device).float()

                    # Compute Q-values and loss 
                    current_q_values = model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
                    next_q_values = target_model(batch_next_state).max(1)[0].detach() 
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

                image_end_time = time()
                image_times.append(image_end_time - image_start_time)
                print(f"Time taken for {current_filename}: {image_end_time - image_start_time:.4f} seconds")

        iteration_end_time = time()
        print(f"Iteration {i} completed in {iteration_end_time - iteration_start_time:.2f} seconds.")