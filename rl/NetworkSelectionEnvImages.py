import os
import re 
import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

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

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.resize_dim),
            transforms.ToTensor(),
        ])

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
        filename = self.dataframe.iloc[self.current_idx]['Filename']
        transformed_image = self.load_image(filename)

        # Convert tensor back to PIL Image to save it
        save_image_path = os.path.join(self.image_folder, f"transformed_{filename}")
        image_to_save = to_pil_image(transformed_image.cpu())  # Convert tensor to PIL Image
        image_to_save.save(save_image_path)  # Save the image
        image_to_save.save('./transformed_image.png')

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

        # Reward is the performance of the selected network
        network_to_normalized_weight = {
            "0%":  0.00,
            "25%": 0.25,
            "50%": 0.50,
            "75%": 0.75
        }

        reward = (selected_network_performance * self.PERFORMANCE_FACTOR) + (1.0 - self.PERFORMANCE_FACTOR) * network_to_normalized_weight[selected_network]

        # Move to the next image
        self.current_idx = (self.current_idx + 1) % len(self.dataframe)
        if self.current_idx == 0:
            done = True
        else:
            done = False

        next_filename = self.dataframe.iloc[self.current_idx]['Filename']
        next_state = self.load_image(next_filename)

        return next_state, reward, done

    def has_next(self):
        """Check if there are more images to process."""
        return self.current_idx < len(self.dataframe) - 1

