import os
import re 
import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from RewardFunctions import linear_factor, logarithmic_factor_40, logarithmic_factor_60, logarithmic_factor_90, dumb_reward_scheme
# from fit_exp_curve import exp_decay, params25, params50, params75, params100
import numpy as np
from fit_polynomial_curve import poly_fit, coefs25, coefs50, coefs75, coefs100


class NetworkSelectionEnvImages:
    def __init__(self, dataframe, image_folder, performance_factor, device, resize_dim=(84, 84), verbose=False):
        self.dataframe = dataframe.copy()

        # Sort by extracting numeric part of 'Filename'
        self.dataframe['sort_key'] = self.dataframe['Filename'].apply(
            lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
        )
        self.image_folder = image_folder
        self.device = device
        self.dataframe = self.dataframe.sort_values('sort_key').reset_index(drop=True)
        self.dataframe.drop('sort_key', axis=1, inplace=True)  # Clean up the temporary column        
        
        self.resize_dim = resize_dim
        self.verbose = verbose
        self.n_actions = 4
        self.current_idx = 0
        self.PERFORMANCE_FACTOR = performance_factor
        self.network_performance_columns = ['0%', '25%', '50%', '75%']

        self.transform = transforms.Compose([
            transforms.Resize(self.resize_dim),
            transforms.ToTensor(),
        ])

        self.battery = 100
        
        # For images
        self.energy_costs = { '0%': 5.28725, '25%': 2.643625, '50%': 1.3218125, '75%': 0.647688125  }

        self.reset()

    def load_image(self, filename):
        """Load and preprocess an image."""
        image_path = os.path.join(self.image_folder, filename)
        image = Image.open(image_path).convert('RGB')  # Ensure RGB
        image_tensor = self.transform(image)
        return self.transform(image).to(self.device)
 
    def reset(self):
        """Point index at the beginning of the dataset (train or test)"""
        self.current_idx = 0
        transformed_image = self.load_image(self.dataframe.iloc[self.current_idx]['Filename'])

        return transformed_image

    def sample_action(self):
        """Randomly select an action."""
        return random.randint(0, self.n_actions - 1)

    def get_next_state_reward(self, action):
        """Determine the next state and reward after taking an action."""
        filename = self.dataframe.iloc[self.current_idx]['Filename']

        performances = {
            '0%':  self.dataframe.iloc[self.current_idx]['0%'],
            '25%': self.dataframe.iloc[self.current_idx]['25%'], 
            '50%': self.dataframe.iloc[self.current_idx]['50%'], 
            '75%': self.dataframe.iloc[self.current_idx]['75%']
        }

        # Sort the networks based on performance in descending order
        sorted_performances = sorted(performances.items(), key=lambda item: item[1], reverse=True)

        selected_network = self.network_performance_columns[action]

        # Extract just the network names from the sorted performances
        network_names_sorted = [network[0] for network in sorted_performances]
        
        if self.verbose:
            print(f"Filename: {filename}")
            print(f"selected_network: {selected_network}")

        # Find the index of the selected network in the sorted list
        selected_network_index = network_names_sorted.index(selected_network)
        if self.verbose:
            print(f"Index of selected network in sorted performances: {selected_network_index}")

        # The selected network's performance field name is assumed to be provided by 'action'
        selected_network_performance = self.dataframe.iloc[self.current_idx][selected_network]

        self.battery -= self.energy_costs[selected_network]

        if self.verbose:        
            print(f"Battery level: {self.battery}")

        # Reward scheme 1: polynomials
        # Calculate reward
        # coefs = {
        #     '0%': coefs100,
        #     '25%': coefs25, # 25% pruned
        #     '50%': coefs50, # 50% pruned
        #     '75%': coefs75, # 75% pruned
        # }

        # reward = np.polyval(coefs[selected_network], self.battery) * 100 + selected_network_performance * 100

        # Reward scheme 2: images left, battery, performance
        # Amount of images left (normalized to [0, 1])
        # images_left = len(self.dataframe) - self.current_idx - 1
        # images_left_normalized = images_left / len(self.dataframe)

        # Battery
        # battery_normalized = (self.battery / 100.0)

        # Energy cost of action
        # max_energy_cost = max(self.energy_costs.values())
        # energy_cost_normalized = self.energy_costs[selected_network] / max_energy_cost
        # efficiency = 1 - energy_cost_normalized

        # performance_weight = battery_normalized * images_left_normalized
        # efficiency_weight = 1 - performance_weight

        # reward = (
        #     performance_weight * selected_network_performance + efficiency_weight * efficiency
        # )

        # Reward scheme 3: weighted sum
        reward = (selected_network_performance * self.PERFORMANCE_FACTOR) + (1.0 - self.PERFORMANCE_FACTOR) * (1 - self.energy_costs[selected_network])


        # Move to the next image
        self.current_idx = (self.current_idx + 1) % len(self.dataframe)
        done = self.current_idx == 0

        next_state = self.load_image(self.dataframe.iloc[self.current_idx]['Filename'])

        if self.battery <= 0:
            return next_state, float('-inf'), done 

        return next_state, reward, done

    def has_next(self):
        """Check if there are more images to process."""
        return self.current_idx < len(self.dataframe) - 1

