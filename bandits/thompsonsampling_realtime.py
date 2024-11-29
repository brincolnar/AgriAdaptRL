import os
import re
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
from Segment import Segment  
from datetime import datetime
import csv

class ContextBanditThompsonGaussian:
    def __init__(self, n_actions, performance_factor, n_features, image_dir, label_dir, model_paths):
        self.n_actions = n_actions
        self.n_features = n_features
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.PERFORMANCE_FACTOR = performance_factor

        self.means = np.zeros(n_actions)
        self.variances = np.ones(n_actions)

        self.features_file = pd.read_csv("./features.csv")
        self.dataset = self.features_file

        self.image_list = sorted(
            [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))],
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
        )        

        print('len(self.image_list)')
        print(len(self.image_list))

        self.feature_columns = [
            'Mean Brightness', 'Hue Hist Feature 1', 'Mean Saturation',
            'Std Brightness', 'Max Brightness', 'Min Brightness',
            'Hue Hist Feature 2', 'Hue Std', 'Contrast', 'Std Saturation',
            'Max Saturation', 'Min Saturation', 'Texture Contrast',
            'Texture Dissimilarity', 'Texture Homogeneity', 'Texture Energy',
            'Texture Correlation', 'Texture ASM', 'Excess Green Index',
            'Excess Red Index', 'CIVE', 'ExG-ExR Ratio', 'CIVE Ratio'
        ]

        all_train_features = self.dataset[self.feature_columns].values
        self.current_image_index = 0
        self.verbose = False
        self.rewards = []
        self.action_to_network = {0: "0%", 1: "25%", 2: "50%", 3: "75%"}
        self.action_to_network_inverse = {v: k for k, v in self.action_to_network.items()}
        self.timing_data = []

        self.segments = {
            i: Segment(model_path=model_paths[i], image_dir=image_dir, label_dir=label_dir, resolution=(512, 512), device="cuda")
            for i in range(n_actions)
        }
        
        self.window = 10000
        self.ts_samples = {i: [] for i in range(self.n_actions)}  
    def write_timing_analysis(self, timing_stats, performance_metrics, output_file="timing_analysis.txt"):
        """
        Write timing analysis statistics and performance metrics to a text file.
        
        Args:
            timing_stats (dict): Dictionary containing timing statistics for each operation
            performance_metrics (dict): Dictionary containing average IoU, weight, and process time
            output_file (str): Path to output file (default: timing_analysis.txt)
        """
        
        with open(output_file, 'w') as f:
            f.write("Performance and Timing Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            from datetime import datetime
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Performance Metrics\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average IoU: {performance_metrics['average_iou']:.4f}\n")
            f.write(f"Average Weight: {performance_metrics['average_weight']:.4f}\n")
            f.write(f"Average Process Time per Image: {performance_metrics['avg_image_time']:.4f} seconds\n\n")
            
            f.write("Detailed Timing Analysis\n")
            f.write("-" * 20 + "\n")
            f.write(f"{'Operation':<20} {'Mean (s)':<15} {'Std Dev (s)':<15} {'Count':<10}\n")
            f.write("-" * 60 + "\n")
            
            total_mean = sum(stats['mean'] for stats in timing_stats.values())
            total_std = (sum(stats['std_dev']**2 for stats in timing_stats.values())) ** 0.5
            
            for operation, stats in timing_stats.items():
                f.write(f"{operation:<20} {stats['mean']:>8.4f}s     {stats['std_dev']:>8.4f}s     {stats['count']:>5}\n")
            
            f.write("-" * 60 + "\n")
            f.write(f"{'Total':<20} {total_mean:>8.4f}s     {total_std:>8.4f}s\n\n")
            
            # Write summary
            f.write("\nTiming Summary:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total processing time: {total_mean:.4f} seconds\n")
            f.write(f"Average time per operation: {total_mean/len(timing_stats):.4f} seconds\n")
            operation_with_max_time = max(timing_stats.items(), key=lambda x: x[1]['mean'])[0]
            f.write(f"Most time-consuming operation: {operation_with_max_time} ({timing_stats[operation_with_max_time]['mean']:.4f}s)\n")

    def analyze_timing_data(self, warmup_images=10):
        """Analyze timing data excluding warmup phase, reporting in seconds."""
        timing_data = self.timing_data[warmup_images:]

        sums = {}
        squares = {}
        counts = {}
		
        # Calculate sums and squared sums
        for data in timing_data:
            for key, value in data.items():
                if key not in sums:
                    sums[key] = 0.0
                    squares[key] = 0.0
                    counts[key] = 0
                sums[key] += value
                squares[key] += value * value
                counts[key] += 1

        # Calculate means and standard deviations
        stats = {}
        for key in sums:
            n = counts[key]
            mean = sums[key] / n
            std_dev = ((squares[key] / n) - (mean * mean)) ** 0.5
            stats[key] = {
                'mean': mean,
                'std_dev': std_dev,
                'count': n
            }

        return stats

    def plot_metrics(self):
        rewards_series = pd.Series(self.rewards)
        smoothed_rewards = rewards_series.rolling(window=self.window).mean()

        plt.figure(figsize=(6, 4))        
        plt.plot(smoothed_rewards, label='Rewards')
        plt.title('Smoothed Rewards over Time')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig('./smoothed_rewards.png')
        plt.close()

    def plot_ts_samples(self):
        plt.figure(figsize=(10, 6))
        for action in range(self.n_actions):
            plt.plot(self.ts_samples[action], label=f'Action {action} ({self.action_to_network[action]})')
        plt.title('Thompson Sampling Values over Time')
        plt.xlabel('Steps')
        plt.ylabel('Sampled Value')
        plt.legend()
        plt.savefig('./thompson_samples_over_time.png')
        plt.close()
    
    def plot_iou(self, ious):
        ious_series = pd.Series(ious)

        window_size = 5  
        smoothed_ious = ious_series.rolling(window=window_size, min_periods=1).mean()

        # IoU values and the smoothed IoU values
        plt.figure(figsize=(12, 6))
        plt.plot(ious_series, label='IoU', alpha=0.3)
        plt.plot(smoothed_ious, label=f'Smoothed IoU (window={window_size})', linewidth=2)
        plt.title('IoU over Time during Training')
        plt.xlabel('Training Steps')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('./thompson_iou.png')
        plt.close()

    def load_image_features(self):
        if self.current_image_index < len(self.dataset):
            row = self.dataset.iloc[self.current_image_index]
            features = row[self.feature_columns].values.astype(float)
            return features, row
        else:
            return None, None

    def select_action(self):
        # Thompson Sampling: sample from the Gaussian distribution for each action
        samples = np.maximum(0, np.random.normal(self.means, np.sqrt(self.variances)))

        # Store sampled values for plotting
        for action in range(self.n_actions):
            self.ts_samples[action].append(samples[action])

        action = np.argmax(samples)  
        return action

    def calculate_reward(self, selected_network, image_name):
        # Use Segment to infer and calculate IoU score for the selected network
        results, _, _, _, timing_data = self.segments[selected_network].infer(image_name)
        selected_network_iou = results['test/iou/weeds']
        
        # Calculate reward based on IoU and performance factor
        network_to_normalized_weight = {"0%": 0.00, "25%": 0.25, "50%": 0.50, "75%": 0.75}
        reward = (selected_network_iou * self.PERFORMANCE_FACTOR) + (1.0 - self.PERFORMANCE_FACTOR) * network_to_normalized_weight[self.action_to_network[selected_network]]
        
        return reward, selected_network_iou, timing_data

    def update(self, action, reward):
        update_time = time()
        # Update the mean and variance using online update rules
        n = len(self.ts_samples[action])  # Number of samples so far

        # Update the mean
        new_mean = (self.means[action] * (n - 1) + reward) / n

        # Update the variance
        new_variance = ((n - 1) * self.variances[action] + (reward - new_mean) * (reward - self.means[action])) / n

        self.means[action] = new_mean
        self.variances[action] = new_variance

        update_time = time() - update_time
        return update_time

    def train(self, epochs):
        start_time = time()  # Start timing the training phase
        correct_selections = 0
        total_iou = 0  
        total_weight = 0 
        ious = []
        image_times = []  

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        csv_filename = f'thompson_{self.PERFORMANCE_FACTOR}_{timestamp}.csv'
        csv_path = os.path.join('./results/thompson/', csv_filename)

        # Energy costs for 457 images
        battery = 100
        energy_costs = { '0%': 0.544, '25%': 0.272, '50%': 0.136, '75%': 0.068 }

        with open(csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Filename', 'Weight', 'Battery Level', 'IoU'])

            # Training
            for epoch in range(epochs):
                for image_name in tqdm(self.image_list, desc="Training Progress"):
                    image_start_time = time()  

                    features, row = self.load_image_features()
                    if features is None:
                        break

                    action = self.select_action(features)

                    reward, iou, timing_data = self.calculate_reward(action, image_name)
                    
                    total_iou += iou
                    ious.append(iou)

                    network_to_weight = {"0%": 100, "25%": 75, "50%": 50, "75%": 25}

                    weight = network_to_weight[self.action_to_network[action]]
                    total_weight += weight

                    battery -= energy_costs[self.action_to_network[action]]

                    csvwriter.writerow([image_name, weight, battery, iou])

                    update_time = self.update(action, reward)
                    timing_data['update'] = update_time
                    
                    self.timing_data.append(timing_data)

                    self.rewards.append(reward)
                    self.current_image_index += 1

                    image_end_time = time() 
                    image_times.append(image_end_time - image_start_time)
                    print(f"Time taken for {image_name}: {image_end_time - image_start_time:.4f} seconds")

                average_iou = total_iou / len(ious)
                average_weight = total_weight / len(ious)
                avg_image_time = sum(image_times) / len(image_times)

                performance_metrics = {
                    'average_iou': average_iou,
                    'average_weight': average_weight,
                    'avg_image_time': avg_image_time
                }

                timing_stats = self.analyze_timing_data()

                # self.write_timing_analysis(timing_stats, performance_metrics, f"./jetson_timing_analysis/thompson_bandit_{self.PERFORMANCE_FACTOR}_timing_analysis.txt")

                print(f"Epoch {epoch+1}: Average IoU: {average_iou:.2f}, Average Weight: {average_weight:.2f}")

            end_time = time()
            avg_image_time = sum(image_times) / len(image_times)
            print(f"Training completed in {end_time - start_time:.2f} seconds.")
            print(f"Average time per image: {avg_image_time:.4f} seconds.")



            # self.plot_iou(ious)
            # self.plot_metrics()
            # self.plot_ts_samples()

if __name__ == "__main__":
    image_dir = "./data/ordered_train_test/all/images"
    label_dir = "./data/ordered_train_test/all/labels"
    model_paths = [
        "./garage/unet_512_pruned_00_iterative_1.pt",
        "./garage/unet_512_pruned_025_iterative_1.pt",
        "./garage/unet_512_pruned_05_iterative_1.pt",
        "./garage/unet_512_pruned_075_iterative_1.pt"
    ]
    performance_factor = float(sys.argv[1])
    epochs = int(sys.argv[2])

    bandit = ContextBanditThompsonGaussian(
        n_actions=4,
        performance_factor=performance_factor,
        n_features=23,
        image_dir=image_dir,
        label_dir=label_dir,
        model_paths=model_paths
    )
    
    print(f"PERFORMANCE FACTOR: {performance_factor}")
    

    bandit.train(epochs)
