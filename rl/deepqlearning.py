import csv
import torch
import random
import os 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

# Assuming action corresponds to the network columns in order
network_performance_columns = [
    '0%', '25%', '50%', '75%'
]

# Create a mapping from action index to network name
action_to_network_mapping = {index: name for index, name in enumerate(network_performance_columns)}

class NetworkSelectionEnv:
    def __init__(self, dataframe, feature_mean, feature_std, verbose):
        self.dataframe = dataframe
        self.n_actions = 4  # Total number of networks/actions
        self.current_idx = -1
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.verbose = verbose
        self.reset()

    def normalize_feature(self, feature):
        """Normalize the feature value."""
        return (feature - self.feature_mean) / self.feature_std

    def reset(self):
        # If we've reached the end of the dataframe, start over
        if self.current_idx >= len(self.dataframe):
            self.current_idx = 0
        return self.normalize_feature(self.dataframe.iloc[self.current_idx][feature])

    def sample_action(self):
        """Randomly select an action."""
        return random.randint(0, self.n_actions - 1)

    def get_next_state_reward(self, action):
        """Determine the next state and reward after taking an action."""
        # Check if the selected action (network) was the best choice
        filename = self.dataframe.iloc[self.current_idx]['Filename']
        # print(f'Filename: {filename}')
        # Get 3 best networks
        performances = {
            '0%':  self.dataframe.iloc[self.current_idx]['0%'],
            '25%': self.dataframe.iloc[self.current_idx]['25%'], 
            '50%': self.dataframe.iloc[self.current_idx]['50%'], 
            '75%': self.dataframe.iloc[self.current_idx]['75%']
        }

        # Sort the networks based on performance in descending order
        sorted_performances = sorted(performances.items(), key=lambda item: item[1], reverse=True)

        selected_network = network_performance_columns[action]
        if self.verbose:
            print(f"selected_network: {selected_network}")

        # Extract just the network names from the sorted performances
        network_names_sorted = [network[0] for network in sorted_performances]
        if self.verbose:
            print("network_names_sorted")
            print(network_names_sorted)

        # Find the index of the selected network in the sorted list
        selected_network_index = network_names_sorted.index(selected_network)
        if self.verbose:
            print(f"Index of selected network in sorted performances: {selected_network_index}")

        # The selected network's performance field name is assumed to be provided by 'action'
        selected_network_performance = self.dataframe.iloc[self.current_idx][selected_network]

         # Reward is the performance of the selected network
        reward = selected_network_performance
        
        reward_bonus = [1.0, 0, -0.5, -1.0]

        # print('reward_bonus[selected_network_index]')
        # print(reward_bonus[selected_network_index])

        reward += reward_bonus[selected_network_index]

        # # Adjust reward based on whether the selected network is in the top 3
        # if action in top_k_networks:
        #     reward += 1  # Add a bonus for selecting a top-3 network
        # else:
        #     reward -= 0  # Penalize for not selecting a top-3 network

        # Move to the next image (this could be random or sequential)
        self.current_idx = (self.current_idx + 1) % len(self.dataframe)

        next_state = self.normalize_feature(self.dataframe.iloc[self.current_idx][feature])

        # Check if the episode is done (if we've gone through all images)
        done = self.current_idx == 0

        return next_state, reward, done

    def has_next(self):
        """Check if there are more images to process."""
        return self.current_idx < len(self.dataframe) - 1


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # Input layer
        self.fc2 = nn.Linear(64, 32) # Hidden layer
        self.fc3 = nn.Linear(32, 4)  # Output layer: Adjust the number of outputs to match your networks

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return output

def evaluate_model(model, test_data, save_to_csv, feature):

    # Put the model in evaluation mode
    model.eval()
    
    test_env = NetworkSelectionEnv(test_df, feature_mean, feature_std, verbose=False)
    
    # Track total reward and actions taken
    total_reward = 0
    correct_predictions = 0
    total_predictions = 0

    state = test_env.reset()
    state = torch.FloatTensor([state]).to(device)

    results = []

    # Loop over the test data
    while test_env.has_next():
        # Move to the next image
        state_tensor = torch.FloatTensor([state]).to(device)

        with torch.no_grad():
            q_values = model(state_tensor).cpu().numpy()
            action = model(state_tensor).argmax().item()

        next_state, reward, done = test_env.get_next_state_reward(action)
        total_reward += reward
        actions_taken.append((test_env.dataframe.iloc[test_env.current_idx]['Filename'], action))

        # Get the best network and selected network performance
        selected_network = network_performance_columns[action]
        best_network = test_env.dataframe.iloc[test_env.current_idx]['best_network']

        print('Filename: ')
        print(test_env.dataframe.iloc[test_env.current_idx]['Filename'])

        print('Selected network: ')
        print(selected_network)

        print('Best network: ')
        print(best_network)

        results.append({
            "Filename": test_env.dataframe.iloc[test_env.current_idx]['Filename'],
            "Predicted Network": selected_network,
            "Best network": best_network
        })

        if selected_network == best_network:
            correct_predictions += 1

        total_predictions += 1

        # IMPORTANT: Update the state with the next state for the next iteration
        state = next_state

    if save_to_csv:
        with open(f'./results/predictions_result_{feature}.csv', 'w', newline='') as file:
            fieldnames = ['Filename', 'Predicted Network', 'Best network']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    return total_reward, accuracy, correct_predictions, total_predictions, actions_taken

def plot_test_split(num_repeat, accuracy):
    plt.figure()

    # Extract the train and test data for the feature and best network performances
    x_train = train_df[feature].tolist()
    x_test = [test_df[test_df['Filename'] == action[0]][feature].values[0] for action in actions_taken]

    best_networks = train_df['best_network'].tolist()
    best_networks_ = test_df['best_network'].tolist()


    print('best_networks_')
    print(best_networks_)
    
    y_train = [train_df.iloc[i][best_network] for i, best_network in enumerate(best_networks)]
    y_test = [test_df[test_df['Filename'] == action[0]][network_performance_columns[action[1]]].values[0] for action in actions_taken]
    truth = [test_df.iloc[i][best_network] for i, best_network in enumerate(best_networks_)]

    # Plot training and testing datasets
    plt.scatter(x_train, y_train, marker='o', color='blue', label='Train')
    plt.scatter(x_test, y_test, marker='h', color='red', label='Test')
    plt.scatter(x_test, truth, marker='x', color='yellow', label='Truth')

    # Adding legend
    plt.legend()

    # Adding title
    plt.title(f'Results for {feature}, Accuracy {round(accuracy, 2)}')

    # Adding labels
    plt.xlabel(feature)
    plt.ylabel('Performance')

    # Save the plot
    plt.savefig(f"./plots/test_plots/{feature.replace(' ', '')}_{num_repeat}_repeat.png")

if __name__ == "__main__":
    
    # Hyperparameters
    episodes = 100 # Number of episodes to train on
    epsilon_start = 1.0  # Starting value of epsilon
    epsilon_end = 0.01  # Minimum value of epsilon
    epsilon_decay = 0.999  # Decay rate of epsilon per episode
    gamma = 0.99  # Discount factor for future rewards
    learning_rate = 0.001  # Learning rate

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss() # nn.SmoothL1Loss()

    # Load the dataset
    features_file = pd.read_csv('./features.csv')
    performance_file = pd.read_csv('./performance_results.csv')

    features = pd.merge(features_file, performance_file, left_on='Filename', right_on='Filename')
    performance_columns = ['0%', '25%', '50%', '75%']
    features['best_network'] = features[performance_columns].idxmax(axis=1)

    train_filenames = os.listdir('./data/geok/train/images')
    test_filenames = os.listdir('./data/geok/test/images')

    # Train-test split
    train_df = features[features['Filename'].isin(train_filenames)]

    all_features = [
    "Mean Brightness", 
    "Std Brightness", 
    "Max Brightness", 
    "Min Brightness",
    "Hue Hist Feature 1",
    "Hue Hist Feature 2",
    "Hue Std",
    "Contrast",
    "Mean Saturation",
    "Std Saturation",
    "Max Saturation",
    "Min Saturation",
    "SIFT Features",
    "Texture Contrast",
    "Texture Dissimilarity",
    "Texture Homogeneity",
    "Texture Energy",
    "Texture Correlation",
    "Texture ASM",
    "Excess Green Index",
    "Excess Red Index",
    "ExG-ExR Ratio",
    "CIVE Ratio"
    ]

    for feature in all_features[-2:]:
        print("========================================================")
        print(f"Evaluating for feature/s: {feature}")
        print("========================================================")

        # Calculate normalization parameters using only the training data
        feature_mean = train_df[feature].mean()
        feature_std = train_df[feature].std()

        test_df = features[features['Filename'].isin(test_filenames)]

        env = NetworkSelectionEnv(train_df, feature_mean, feature_std, verbose=False)

        num_iterations = 10

        avg_accuracy = 0

        for i in range(num_iterations):
            losses = []

            print("========================================================")
            print(f"Iteration {i} starting...")
            print("========================================================")

            for episode in tqdm(range(episodes), desc=f'Iteration {i}'):
                state = env.reset()  # Reset the environment
                state = torch.FloatTensor([state]).to(device)

                epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))  # Decay epsilon
                
                for t in range(1000):
                    # Select action
                    if random.random() > epsilon:
                        with torch.no_grad():
                            action = model(state).argmax().item()
                    else:
                        action = env.sample_action()
                    
                    # print(f'action: {action}')

                    # Execute action
                    next_state, reward, done = env.get_next_state_reward(action)
                    next_state = torch.FloatTensor([next_state]).to(device)
                    # print(f'next_state: {next_state}')

                    # Compute target and loss
                    target = reward + gamma * model(next_state).max().item() * (not done)
                    target = torch.tensor([target], device=device, dtype=torch.float) 
                    prediction = model(state)[action]
                    # print(f'target: {target}')
                    # print(f'prediction: {prediction}')
                    loss = loss_fn(prediction, target.unsqueeze(0))  # Adjust target shape if necessary
                    losses.append(loss)

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    q_values = model(state)
                    # print('q_values')
                    # print(q_values)
                    
                    if done:
                        break
                    state = next_state

            loss_values = [loss.item() for loss in losses]  # Convert each loss tensor to a scalar
            plt.figure(figsize=(10,5))
            plt.plot(loss_values, label='Loss')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f"./plots/losses/training_loss_plot_feature_{feature.replace(' ', '')}_repeat_{i}.png", bbox_inches='tight')

            actions_taken = []

            # Only write last iteration
            save_to_csv = i == 9

            # Evaluate the model
            total_test_reward, accuracy, correct_predictions, total_predictions, actions_taken = evaluate_model(model, test_df, save_to_csv=save_to_csv, feature=feature)
            print(f"Total test reward: {total_test_reward}")

            # Collect results
            for filename, action in actions_taken:
                print(f"Filename: {filename}, Predicted Network: {action_to_network_mapping[action]}")

            print(f"Accuracy: {accuracy:.2f}")
            print(f"Correct Predictions: {correct_predictions}, Total Predictions: {total_predictions}")

            avg_accuracy += accuracy

            plot_test_split(i, accuracy)

        print("----------------------------------------------------------")
        print(f"Average accuracy: {avg_accuracy / num_iterations}")
        print(f"Feature used: {feature}")
        print("----------------------------------------------------------")
