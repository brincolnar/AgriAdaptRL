# Implement a context bandit which uses neural network for predicting rewards
import os
import re
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm 

class NeuralBanditModel(nn.Module):
    def __init__(self, n_features, n_actions):
        super(NeuralBanditModel, self).__init__()

        # Network 1
        # self.fc = nn.Sequential(
        #     nn.Linear(n_features, 128),  # Input layer
        #     nn.ReLU(),
        #     nn.Linear(128, n_actions)    # Output layer with one output per action
        # )

        # Network 2
        # self.fc = nn.Sequential(
        #     nn.Linear(n_features, 256),  
        #     nn.ReLU(),
        #     nn.Linear(256, 128),        
        #     nn.ReLU(),
        #     nn.Linear(128, n_actions)
        # )

        # Network 3 (4 layers, more neurons and layers than Network 2)
        self.fc = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        # Network 4 (5 layers, more neurons and layers than Network 3)
        # self.fc = nn.Sequential(
        #     nn.Linear(n_features, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, n_actions)
        # )
    
    def forward(self, x):
        return self.fc(x)

class ContextBandit:
    def __init__(self, n_actions, n_features, image_dir, performance_factor, learning_rate=0.001, epsilon=0.1):
        self.n_actions = n_actions
        self.n_features = n_features
        self.window = 10000
        self.epsilon = epsilon
        self.model = NeuralBanditModel(n_features, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.scaler = StandardScaler()
        self.image_dir = image_dir
        self.features_file = pd.read_csv("./features.csv")
        self.performance_file = pd.read_csv('./performance_results.csv')
        self.dataset = pd.merge(self.features_file, self.performance_file, on='Filename')
        self.PERFORMANCE_FACTOR = performance_factor
        
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

    def plot_metrics(self):

        rewards_series = pd.Series(self.rewards)
        smoothed_rewards = rewards_series.rolling(window=self.window).mean()

        plt.figure(figsize=(6, 4))        
        plt.plot(smoothed_rewards, label='Rewards')
        plt.title('Smoothed Rewards over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig('./smoothed_rewards.png')
        plt.close()

        prediction_series = pd.Series(self.prediction_errors)
        smoothed_predictions = prediction_series.rolling(window=self.window).mean()

        plt.figure(figsize=(6, 4))
        plt.plot(self.prediction_errors, label='Prediction Errors', color='red')
        plt.title('Prediction Errors over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Prediction Error')
        plt.legend()
        plt.savefig('./smoothed_prediction_errors.png')
        plt.close() 

    def predict_rewards(self, context):
        context = torch.FloatTensor(context).unsqueeze(0)  # Convert context to tensor and add batch dimension
        with torch.no_grad():
            predicted_rewards = self.model(context)
        return predicted_rewards.numpy().squeeze(0)  

    def fit(self, context, action, reward):
        self.model.train()
        context = torch.FloatTensor(context).unsqueeze(0)
        predictions = self.model(context)
        
        target = predictions.clone()
        target[0, action] = reward
        prediction_error = (predictions[0, action] - reward).item()
        self.prediction_errors.append(prediction_error) 

        loss = self.loss_fn(predictions, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return self.select_action(context.numpy().squeeze(0))

    def load_image_features(self):
        if self.current_image_index < len(self.train_df):
            row = self.train_df.iloc[self.current_image_index]
            features = row[self.feature_columns].values.astype(float)
            return features
        else:
            return None

    def select_action(self, context):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            predicted_rewards = self.predict_rewards(context)
            action = np.argmax(predicted_rewards)
        return action

    def calculate_reward(self, selected_network):
        selected_network_name = self.action_to_network[selected_network]
        selected_network_performance = self.train_df.iloc[self.current_image_index][selected_network_name]
        network_to_normalized_weight = {"0%": 0.00, "25%": 0.25, "50%": 0.50, "75%": 0.75}
        reward = (selected_network_performance * self.PERFORMANCE_FACTOR) + (1.0 - self.PERFORMANCE_FACTOR) * network_to_normalized_weight[selected_network_name]
        return reward

    def calculate_test_reward(self, row, selected_network):
        selected_network_name = self.action_to_network[selected_network]
        selected_network_performance = row[selected_network_name]
        network_to_normalized_weight = {"0%": 0.00, "25%": 0.25, "50%": 0.50, "75%": 0.75}
        reward = (selected_network_performance * self.PERFORMANCE_FACTOR) + (1.0 - self.PERFORMANCE_FACTOR) * network_to_normalized_weight[selected_network_name]
        return reward
    
    def train(self, epochs):
        # Training
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            while True:
                features = self.load_image_features()
                if features is None:
                    self.current_image_index = 0
                    break
                action = bandit.select_action(features)
                reward = bandit.calculate_reward(action)
                self.rewards.append(reward)
                next_action = bandit.fit(features, action, reward)
                bandit.current_image_index += 1  
        self.plot_metrics()

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
            
            # Use the neural network model to predict rewards
            predicted_rewards = self.predict_rewards(features)            
            selected_action = np.argmax(predicted_rewards)  

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

    bandit = ContextBandit(n_actions=4, n_features=23, image_dir=image_dir, performance_factor=performance_factor)

    # Training
    bandit.train(epochs)

    # Testing
    bandit.test()