import os 
import numpy as np
import pandas as pd
from WidthAdjustableSingleImageInference import WidthAdjustableSingleImageInference
from metaseg_eval import compute_metrics_i
from metaseg_io import probs_gt_save

# Utility functions
def get_state_from_image(row):

    # Extract the feature values
    max_saturation_interval = row['Max Saturation Interval']
    hue_hist_feature_2_interval = row['Hue Hist Feature 2 Interval']
    mean_brightness_interval = row['Mean Brightness Interval']

    # Map the combination of features to a unique state
    # Considering the combination as a 3-digit number in base-3
    state = max_saturation_interval * 9 + hue_hist_feature_2_interval * 3 + mean_brightness_interval

    return state

def apply_neural_network_and_get_reward(action,row,episode,train):
    print("action")
    print(action)

    width_mapping = {0: 0.25, 1: 0.5, 2: 0.75, 3: 1.0}
    width = width_mapping[action]

    print("width")
    print(width)

    inference_engine = WidthAdjustableSingleImageInference(
        dataset='geok',
        image_resolution=(256, 256),
        model_architecture="squeeze",
        width=width,
        filename=row['Filename'],
        train=train,
        save_image=True,
        is_trans=True,
        is_best_fitting=False,
    )

    results, probs, gt, filename = inference_engine.infer(row['Filename'])

    use_iou = True

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

features = pd.read_csv("./features.csv")
train_data = os.listdir("./data/geok/train/images/")
train_data.sort()

q_table = np.zeros((27, 4))

# Q(state,action)=Q(state,action)+α×reward
alpha = 0.3
epsilon = 0.2
epsilon_max = epsilon
epsilon_dropout = 0.04

num_repeats = 10
for i in range(num_repeats):
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

        print('reward: ')
        print(reward)

        # Q-learning update
        q_table[state, action] = q_table[state, action] + alpha * reward

        # Update epsilon
        # epsilon = update_epsilon(epsilon)

print("===========================Training finished===========================")

print("q_table: ")
print(q_table)

test_data = os.listdir("./data/geok/test/images/")
test_data.sort()

# Accumulated iou
iou = 0

for episode in range(len(test_data)): 
    row = features[features['Filename'] == test_data[episode]].iloc[0]

    state = get_state_from_image(row)

    action = np.argmax(q_table[state, :])

    # Apply selected neural network and get reward
    _, iou_ = apply_neural_network_and_get_reward(action, row, episode, train=False)
    
    # Add to accumulative iou
    iou += iou_

print("iou:")
print(iou)
