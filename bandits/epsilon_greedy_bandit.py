import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm 

'''
Class for training, testing contextual bandit using linear 
'''
class ContextBandit:
    def __init__(self, n_actions, n_features, image_dir, performance_factor, epochs, learning_rate=0.01, epsilon=0.10):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.window = 100000
        self.epsilon = epsilon
        self.weights = np.random.normal(size=(n_actions, n_features))
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
        # all_train_features = self.train_df[self.feature_columns].values
        all_features = self.dataset[self.feature_columns].values

        self.scaler = StandardScaler().fit(all_features)
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

    def load_image_features(self):
        if self.current_image_index < len(self.dataset):
            row = self.dataset.iloc[self.current_image_index]
            features = row[self.feature_columns].values.astype(float)
            return features, row
        else:
            return None, None

    def predict_rewards(self, context):
        context = self.scaler.transform([context])[0]
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
        context = self.scaler.transform([context])[0]
        prediction = np.dot(self.weights[action], context)
        error = reward - prediction

        self.prediction_errors.append(abs(error)) 
        self.weights[action] += self.learning_rate * error * context

        self.rewards.append(reward) 
        return self.select_action(context)

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
    
    def train(self, epochs):
        # Training

        total_actions_taken = np.zeros(self.n_actions)
        correct_selections = 0
        total_iou = 0  
        total_weight = 0 
        true_actions = []
        total_reward = 0

        for epoch in tqdm(range(epochs), desc="Training Progress"):
            while True:
                features, row = self.load_image_features()
                if features is None:
                    break
                action = bandit.select_action(features)

                selected_network_name = self.action_to_network[action]
                selected_network_iou = row[selected_network_name]
                total_iou += selected_network_iou
            
                true_action = self.action_to_network_inverse[row["best_network"]]
                true_actions.append(true_action) 

                if action == true_action:
                    correct_selections += 1

                network_to_weight = {"0%": 100, "25%": 75, "50%": 50, "75%": 25}
                total_weight += network_to_weight[selected_network_name]

                reward = bandit.calculate_reward(action)
                next_action = bandit.fit(features, action, reward)
                bandit.current_image_index += 1  
            
            average_iou = total_iou / len(self.dataset) 
            average_weight = total_weight / len(self.dataset) 
            accuracy = correct_selections / len(self.dataset)

            print(f"Average IoU: {average_iou:.2f}")
            print(f"Average Weight: {average_weight:.2f}")
            print(f"Accuracy: {accuracy * 100:.2f}%")

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
            predicted_rewards = np.dot(self.weights, features)  
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

        return total_actions_taken, accuracy


if __name__ == "__main__":
    image_dir = "data/ordered_train_test"
    performance_factor = float(sys.argv[1])
    epochs = int(sys.argv[2])

    bandit = ContextBandit(n_actions=4, n_features=23, image_dir=image_dir, performance_factor=performance_factor, epochs=epochs)

    # Training
    bandit.train(epochs)

    # Testing
    # bandit.test()