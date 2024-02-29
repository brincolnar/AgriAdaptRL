import os 
import numpy as np
import pandas as pd
from measure_pruned0_preformance import infer0
from measure_pruned025_preformance import infer025
from measure_pruned05_preformance import infer05
from measure_pruned075_preformance import infer075
from metaseg_eval import compute_metrics_i
from metaseg_io import probs_gt_save
import matplotlib.pyplot as plt

# Utility functions
def get_state_from_image(row):

    # Extract the feature values
    feature1 = row['Hue Std Interval'] - 1
    feature2 = row['Contrast Interval'] - 1 
    feature3 = row['CIVE Interval'] - 1

    # Map the combination of features to a unique state
    # Considering the combination as a 3-digit number in base-3
    state = feature1 * 16 + feature2 * 4 + feature3

    return state


energy_usage = {
    0: 1.0,   
    1: 0.75,  
    2: 0.5, 
    3: 0.25   
}

performance_weight = 0.7 
energy_weight = 1.0 - performance_weight

def apply_neural_network_and_get_reward(action,row,episode,train):
    print("action")
    print(action)

    if action == 0:
        results, probs, gt, filename = infer0(row['Filename'])
    elif action == 1:
        results, probs, gt, filename = infer025(row['Filename'])
    elif action == 2:
        results, probs, gt, filename = infer05(row['Filename'])
    elif action == 3:
        results, probs, gt, filename = infer075(row['Filename'])

    use_iou = True

    if use_iou:
        base_iou_score = results['test/iou/weeds']
    else:
        base_iou_score = calculate_reward(probs, gt, row['Filename'], episode)

    energy_efficiency_score = 1 - energy_usage[action]  
    final_reward = (base_iou_score * performance_weight) + (energy_efficiency_score * energy_weight)

    return final_reward, results['test/iou/weeds']


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

# 4**3 possible intervalized feature states, 4 possible network configs
q_table = np.zeros((64, 4))

# Q(state,action)=Q(state,action)+α×reward
alpha = 0.7
epsilon_start = 0.9
epsilon_min = 0.1
epsilon_decay = 0.995  # Adjust based on the total number of episodes planned

num_repeats = 100
episode_rewards = [] 

epsilon = epsilon_start
for i in range(num_repeats):
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    for episode in range(len(train_data)):
        # Find the row in the DataFrame where the Filename matches image_path
        row = features[features['Filename'] == train_data[episode]].iloc[0]

        state = get_state_from_image(row)
        if np.random.uniform(0, 1) < epsilon:
            # Exploration
            action = np.random.choice(4)
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