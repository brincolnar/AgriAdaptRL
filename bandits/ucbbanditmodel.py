import os
import re
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class ContextBanditUCB:
    def __init__(self, n_actions, performance_factor, n_features, image_dir, c=1):
        self.n_actions = n_actions
        self.n_features = n_features
        self.image_dir = image_dir
        self.c = c  # Exploration parameter for UCB
        self.weights = np.random.normal(size=(n_actions, n_features))
        self.action_counts = np.zeros(n_actions)  # Count of selections for each action
        self.sum_rewards = np.zeros(n_actions)    # Sum of rewards for each action
        self.average_rewards = np.zeros(n_actions)
        self.total_iterations = 0
        self.scaler = StandardScaler()
        self.features_file = pd.read_csv("./features.csv")
        self.performance_file = pd.read_csv('./performance_results.csv')
        self.dataset = pd.merge(self.features_file, self.performance_file, on='Filename')
        self.PERFORMANCE_FACTOR = performance_factor
        self.learning_rate = 0.01

        # Assign the best network based on performance metrics (for evaluation)
        performance_columns = ['0%', '25%', '50%', '75%']
        self.dataset['best_network'] = self.dataset[performance_columns].idxmax(axis=1)

        train_filenames = os.listdir(f'../{self.image_dir}/train/images')
        test_filenames = os.listdir(f'../{self.image_dir}/test/images')

        # Sort by extracting numeric part of 'Filename'
        self.dataset['sort_key'] = self.dataset['Filename'].apply(
            lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
        )
        self.dataset = self.dataset.sort_values('sort_key').reset_index(drop=True)
        self.dataset.drop('sort_key', axis=1, inplace=True)  

        
        self.train_df = self.dataset[self.dataset['Filename'].isin(train_filenames)]
        self.test_df = self.dataset[self.dataset['Filename'].isin(test_filenames)]

        self.feature_columns = [
            'Mean Brightness', 'Hue Hist Feature 1', 'Mean Saturation',
            'Std Brightness', 'Max Brightness', 'Min Brightness',
            'Hue Hist Feature 2', 'Hue Std', 'Contrast', 'Std Saturation',
            'Max Saturation', 'Min Saturation', 'Texture Contrast',
            'Texture Dissimilarity', 'Texture Homogeneity', 'Texture Energy',
            'Texture Correlation', 'Texture ASM', 'Excess Green Index',
            'Excess Red Index', 'CIVE', 'ExG-ExR Ratio', 'CIVE Ratio'
        ]

        # Fit the scaler on all feature data at once
        all_train_features = self.train_df[self.feature_columns].values
        self.scaler = StandardScaler().fit(all_train_features)
        self.current_image_index = 0
        self.verbose = False
        self.rewards = []
        self.prediction_errors = []
        self.action_to_network = {0: "0%", 1: "25%", 2: "50%", 3: "75%"}
        self.action_to_network_inverse = {v: k for k, v in self.action_to_network.items()}

        # for plotting
        self.window = 10000
        self.ucb_values = {i: [] for i in range(self.n_actions)} 

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

    def plot_ucb_values(self):
        plt.figure(figsize=(10, 6))
        for action in range(self.n_actions):
            plt.plot(self.ucb_values[action], label=f'Action {action} ({self.action_to_network[action]})')
        plt.title('UCB Values over Time')
        plt.xlabel('Steps')
        plt.ylabel('UCB Value')
        plt.legend()
        plt.savefig('./ucb_values_over_time.png')
        plt.close()

    def load_image_features(self):
        if self.current_image_index < len(self.dataset):
            row = self.dataset.iloc[self.current_image_index]
            features = row[self.feature_columns].values.astype(float)
            return features, row
        else:
            return None, None

    def select_action(self, context):

        scaled_context = self.scaler.transform([context])
        print(scaled_context)
        
        # Calculate UCB values for each action
        ucb_values = self.average_rewards + self.c * np.sqrt(np.log(self.total_iterations + 1) / (self.action_counts + 1))

        # Store UCB values for plotting
        for action in range(self.n_actions):
            self.ucb_values[action].append(ucb_values[action])        
        
        action = np.argmax(ucb_values)
        print(action)
        return action

    def calculate_reward(self, selected_network):
        selected_network_name = self.action_to_network[selected_network]
        selected_network_performance = self.dataset.iloc[self.current_image_index][selected_network_name]
        network_to_normalized_weight = {"0%": 0.00, "25%": 0.25, "50%": 0.50, "75%": 0.75}
        reward = (selected_network_performance * self.PERFORMANCE_FACTOR) + (1.0 - self.PERFORMANCE_FACTOR) * network_to_normalized_weight[selected_network_name]
        return reward

    def calculate_test_reward(self, row, selected_network):
        selected_network_name = self.action_to_network[selected_network]
        selected_network_performance = row[selected_network_name]
        network_to_normalized_weight = {"0%": 0.00, "25%": 0.25, "50%": 0.50, "75%": 0.75}
        reward = (selected_network_performance * self.PERFORMANCE_FACTOR) + (1.0 - self.PERFORMANCE_FACTOR) * network_to_normalized_weight[selected_network_name]
        return reward

    def update(self, action, reward, context):
        self.action_counts[action] += 1
        self.sum_rewards[action] += reward
        self.average_rewards[action] = self.sum_rewards[action] / self.action_counts[action]
        self.total_iterations += 1

        # Update weights using linear regression update rule
        prediction = np.dot(self.weights[action], context)
        error = reward - prediction
        self.weights[action] += self.learning_rate * error * context

    def train(self, epochs):
        correct_selections = 0
        total_iou = 0  
        total_weight = 0 
        true_actions = []

        # Training
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            while True:
                features, row = self.load_image_features()
                if features is None:
                    self.current_image_index = 0
                    break

                action = bandit.select_action(features)

                selected_network_name = self.action_to_network[action]
                select_network_iou = row[selected_network_name]
                total_iou += select_network_iou

                true_action = self.action_to_network_inverse[row["best_network"]]
                true_actions.append(true_action) 

                if action == true_action:
                    correct_selections += 1

                network_to_weight = {"0%": 100, "25%": 75, "50%": 50, "75%": 25}
                total_weight += network_to_weight[selected_network_name]

                reward = bandit.calculate_reward(action)
                self.rewards.append(reward)
                self.update(action, reward, features)
                bandit.current_image_index += 1  

            average_iou = total_iou / len(self.dataset) 
            average_weight = total_weight / len(self.dataset) 
            accuracy = correct_selections / len(self.dataset)
        
            print(f"Average IoU:    {average_iou:.2f}")
            print(f"Average Weight: {average_weight:.2f}")
            print(f"Accuracy: {accuracy * 100:.2f}%")

        self.plot_metrics()
        self.plot_ucb_values() 

    def test(self):
        print("Starting testing phase...")
        total_actions_taken = np.zeros(self.n_actions)
        correct_selections = 0
        total_iou = 0  
        total_weight = 0 
        true_actions = []
        total_reward = 0

        for index, row in self.test_df.iterrows():
            features = row[self.feature_columns].values.astype(float)
            features = self.scaler.transform([features])[0]
            
            selected_action = self.select_action(features)  # Use the UCB action selection directly

            true_action = self.action_to_network_inverse[row["best_network"]]
            true_actions.append(true_action) 

            reward = self.calculate_test_reward(row, selected_action) 
            total_reward += reward
            total_actions_taken[selected_action] += 1

            if selected_action == true_action:
                correct_selections += 1

            selected_network_name = self.action_to_network[selected_action]
            selected_network_iou = row[selected_network_name]
            total_iou += selected_network_iou
            
            network_to_weight = {"0%": 100, "25%": 75, "50%": 50, "75%": 25}
            total_weight += network_to_weight[selected_network_name]

        total_tests = len(self.test_df)
        average_iou = total_iou / total_tests if total_tests > 0 else 0
        average_weight = total_weight / total_tests if total_tests > 0 else 0
        accuracy = correct_selections / total_tests if total_tests > 0 else 0
        
        print("Testing completed.")
        print(f"Actions Taken: {total_actions_taken}")
        print(f"True Actions: {true_actions}")
        print(f"Average IoU: {average_iou:.2f}")
        print(f"Average Weight: {average_weight:.2f}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Total Reward: {total_reward}")
        print(f"Performance factor: {self.PERFORMANCE_FACTOR}")

        return total_actions_taken, accuracy


if __name__ == "__main__":
    image_dir = "data/ordered_train_test"
    performance_factor = float(sys.argv[1])
    epochs = int(sys.argv[2])

    bandit = ContextBanditUCB(n_actions=4, performance_factor=performance_factor, n_features=23, image_dir=image_dir, c=1)

    # Training
    bandit.train(epochs)

    # Testing
    # bandit.test()