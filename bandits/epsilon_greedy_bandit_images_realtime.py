import os
import re
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from torchvision import transforms
from Segment import Segment  
from PIL import Image
import csv
from datetime import datetime


class ContextBandit:
    def __init__(self, n_actions, performance_factor, image_dir, label_dir, model_paths, epochs, resize_dim=(84, 84), inference_dim=(512, 512), learning_rate=0.01, epsilon=0.10):
        self.n_actions = n_actions
        self.n_features = resize_dim[0] * resize_dim[1] * 3
        self.learning_rate = learning_rate
        self.window = 100000
        self.epsilon = epsilon
        self.weights = np.random.normal(size=(n_actions, self.n_features))
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.PERFORMANCE_FACTOR = performance_factor
        self.current_image_index = 0
        self.rewards = []
        self.prediction_errors = []
        self.action_to_network = {0: "0%", 1: "25%", 2: "50%", 3: "75%"}
        self.action_to_network_inverse = {v: k for k, v in self.action_to_network.items()}
        self.timing_data = []

        self.segments = {
            i: Segment(model_path=model_paths[i], image_dir=image_dir, label_dir=label_dir, resolution=inference_dim, device="cuda")
            for i in range(n_actions)
        }

        self.image_list = sorted(
            [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))],
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
        )

        self.feature_transform = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.ToTensor()
        ])

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
		
        for data in timing_data:
            for key, value in data.items():
                if key not in sums:
                    sums[key] = 0.0
                    squares[key] = 0.0
                    counts[key] = 0
                sums[key] += value
                squares[key] += value * value
                counts[key] += 1

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

    def load_image_features(self, image_name):
        """
        Load and downscale the image to 84x84 for feature extraction.
        """
        image_path = os.path.join(self.image_dir, image_name)
        img = Image.open(image_path).convert('RGB')
        img = self.feature_transform(img)
        features = img.flatten().numpy()  
        return features

    def predict_rewards(self, context):
        predicted_rewards = np.dot(self.weights, context)
        return predicted_rewards

    def select_action(self, context):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            predicted_rewards = self.predict_rewards(context)
            action = np.argmax(predicted_rewards)
        return action

    def fit(self, context, action, reward):
        fit_time = time()
        prediction = np.dot(self.weights[action], context)
        error = reward - prediction
        self.prediction_errors.append(abs(error))
        self.weights[action] += self.learning_rate * error * context
        self.rewards.append(reward)
        fit_time = time() - fit_time
        return self.select_action(context), fit_time

    def calculate_reward(self, selected_network, image_name):
        results, _, _, _, timing_data = self.segments[selected_network].infer(image_name)
        selected_network_iou = results['test/iou/weeds']
        
        network_to_normalized_weight = {"0%": 0.00, "25%": 0.25, "50%": 0.50, "75%": 0.75}
        reward = (selected_network_iou * self.PERFORMANCE_FACTOR) + (1.0 - self.PERFORMANCE_FACTOR) * network_to_normalized_weight[self.action_to_network[selected_network]]
        
        return reward, selected_network_iou, timing_data

    def train(self, epochs):
        start_time = time()  
        total_iou = 0
        total_weight = 0
        correct_selections = 0
        ious = []
        image_times = []  
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # include timestamp for uniqueness
        csv_filename = f'epsilon_{self.PERFORMANCE_FACTOR}_{timestamp}.csv'
        csv_path = os.path.join('./results/epsilon/', csv_filename)

        battery = 100

        # For 457 images
        energy_costs = { '0%': 0.544, '25%': 0.272, '50%': 0.136, '75%': 0.068 }

        with open(csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Filename', 'Weight', 'Battery Level', 'IoU'])

            for epoch in range(epochs):
                for image_name in tqdm(self.image_list, desc="Training Progress"):
                    image_start_time = time()  
                    

                    features = self.load_image_features(image_name) 
                    action = self.select_action(features)

                    reward, iou, timing_data = self.calculate_reward(action, image_name)

                    # print(f'image_name: {image_name}, iou: {iou}')

                    total_iou += iou
                    ious.append(iou)

                    network_to_weight = {"0%": 100, "25%": 75, "50%": 50, "75%": 25}

                    weight = network_to_weight[self.action_to_network[action]]
                    total_weight += weight

                    battery -= energy_costs[self.action_to_network[action]]

                    csvwriter.writerow([image_name, weight, battery, iou])

                    _, fit_time = self.fit(features, action, reward)
                    
                    timing_data['fit'] = fit_time
                    
                    self.timing_data.append(timing_data)

                    image_end_time = time() 
                    image_times.append(image_end_time - image_start_time)
                    # print(f"Time taken for {image_name}: {image_end_time - image_start_time:.4f} seconds")

                timing_stats = self.analyze_timing_data()

                average_iou = total_iou / len(ious)
                average_weight = total_weight / len(ious)
                avg_image_time = sum(image_times) / len(image_times)

                print(f"Epoch {epoch+1}: Average IoU: {average_iou:.2f}, Average Weight: {average_weight:.2f}")

                performance_metrics = {
                    'average_iou': average_iou,
                    'average_weight': average_weight,
                    'avg_image_time': avg_image_time
                }

                # self.write_timing_analysis(timing_stats, performance_metrics, f"./jetson_timing_analysis/epsilon_bandit_{self.PERFORMANCE_FACTOR}_timing_analysis.txt")

        end_time = time()
        avg_image_time = sum(image_times) / len(image_times)
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
        print(f"Average time per image: {avg_image_time:.4f} seconds.")


if __name__ == "__main__":
    image_dir = "./data/ordered_train_test/all/images"        # Directory containing images for segmentation
    label_dir = "./data/ordered_train_test/all/labels"        # Directory containing ground truth labels
    model_paths = [
        "./garage/unet_512_pruned_00_iterative_1.pt",          # Path to unpruned model
        "./garage/unet_512_pruned_025_iterative_1.pt",         # Path to 25% pruned model
        "./garage/unet_512_pruned_05_iterative_1.pt",          # Path to 50% pruned model
        "./garage/unet_512_pruned_075_iterative_1.pt"          # Path to 75% pruned model
    ]
    performance_factor = float(sys.argv[1])
    epochs = int(sys.argv[2])

    bandit = ContextBandit(
        n_actions=4,
        performance_factor=performance_factor,
        image_dir=image_dir,
        label_dir=label_dir,
        model_paths=model_paths,
        epochs=epochs
    )
    
    print(f"PERFORMANCE FACTOR: {performance_factor}")

    bandit.train(epochs)
