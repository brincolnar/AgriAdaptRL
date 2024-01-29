from stable_baselines3 import PPO
from modelselectionenv import ModelSelectionEnv
from PIL import Image
import numpy as np
from torchvision import transforms
from spectral_features import SpectralFeatures
import os
from metaseg_eval import compute_metrics_i

# Create a list of all train image paths
# image_dir = './data/geok/train/images/'
image_dir = './small_data/geok/train/images/'
image_paths = [image_dir + img for img in os.listdir(image_dir)]

# Create and train the RL agent
env = ModelSelectionEnv(image_paths)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1)

# Save the trained model
model.save("./models/ppo_uav_model")

