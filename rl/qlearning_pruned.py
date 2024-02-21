import os 
import numpy as np
import pandas as pd
from measure_pruned025_preformance import infer025
from measure_pruned05_preformance import infer05
from measure_pruned075_preformance import infer075
from metaseg_eval import compute_metrics_i
from metaseg_io import probs_gt_save
import matplotlib.pyplot as plt

# Utility functions
def get_state_from_image(row):

    # Extract the feature values
    hue_std_interval = row['Hue Std Interval'] - 1
    contrast_interval = row['Contrast Interval'] - 1 
    sift_features_interval = row['SIFT Features Interval'] - 1

    # Map the combination of features to a unique state
    # Considering the combination as a 3-digit number in base-3
    state = hue_std_interval * 9 + contrast_interval * 3 + sift_features_interval

    return state


def apply_neural_network_and_get_reward(action,row,episode,train):
    print("action")
    print(action)

    width_mapping = {0: 0.25, 1: 0.5, 2: 0.75}

    if action == 0:
        results, probs, gt, filename = infer025(row['Filename'])
    elif action == 1:
        results, probs, gt, filename = infer05(row['Filename'])
    else:
        results, probs, gt, filename = infer075(row['Filename'])

    use_iou = False

    if use_iou:
        return results['test/iou/weeds'], results['test/iou/weeds'] 
    else:
        return calculate_reward(probs, gt, row['Filename'], episode), results['test/iou/weeds']

def calculate_reward(probs, gt, filename, episode):
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    probs_softmax = softmax(probs)
    probs_gt_save(probs_softmax, gt, filename, episode)
    S = compute_metrics_i(episode)

    print(f"S: {S}")

    return S

def update_epsilon(epsilon):
    return epsilon - epsilon_max * epsilon_dropout

features = pd.read_csv("./cleaned_features.csv")
train_data = os.listdir("./data/geok/train/images/")
train_data.sort()

# 27 possible intervalized feature states, 3 possible network configs
q_table = np.zeros((27, 3))

# Q(state,action)=Q(state,action)+α×reward
alpha = 0.3
epsilon = 0.2
epsilon_max = epsilon
epsilon_dropout = 0.04

num_repeats = 10
episode_rewards = [] 

for i in range(num_repeats):
    for episode in range(len(train_data)):
        # Find the row in the DataFrame where the Filename matches image_path
        row = features[features['Filename'] == train_data[episode]].iloc[0]

        state = get_state_from_image(row)
        if np.random.uniform(0, 1) < epsilon:
            # Exploration
            action = np.random.choice(3)
        else:
            # Exploitation
            action = np.argmax(q_table[state, :])

        # Apply selected neural network and get reward
        reward, _ = apply_neural_network_and_get_reward(action, row, episode, train=True)
        episode_rewards.append(reward)  # Store the reward for plotting

        print('reward: ')
        print(reward)

        # Q-learning update
        q_table[state, action] = q_table[state, action] + alpha * reward

        # Update epsilon
        # epsilon = update_epsilon(epsilon)

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label='Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward per Episode Over Time')
plt.legend()

# Saving the plot as an image file
plt.savefig('episode_rewards.png', dpi=300)  # Save as PNG with high resolution

print("===========================Training finished===========================")

print("q_table: ")
print(q_table)