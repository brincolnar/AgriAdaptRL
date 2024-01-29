import gym
from gym import spaces
import numpy as np


import os
from pathlib import Path
from random import randint

from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks

from segmentation import settings
from segmentation.data.data import ImageImporter
from segmentation.helpers.metricise import Metricise
from adaptation.inference import AdaptiveWidth
from segmentation.models.slim_squeeze_unet import SlimSqueezeUNetCofly, SlimSqueezeUNet
from segmentation.models.slim_unet import SlimUNet
from segmentation.helpers.masking import get_binary_masks_infest
from segmentation.data.data import ImageImporter
from metaseg_io import probs_gt_save

import numpy as np # for softmax

from spectral_features import SpectralFeatures
from metaseg_eval import compute_metrics_i

class WidthAdjustableSingleImageInference:
    def __init__(
        self,
        dataset,
        image_resolution,
        model_architecture,
        width,
        filename,
        fixed_image=-1,
        save_image=False,
        is_trans=False,
        is_best_fitting=False,
    ):
        self.project_path = Path(settings.PROJECT_DIR)
        if dataset == "infest":
            self.image_dir = "segmentation/data/agriadapt/NN_labeled_samples_salad_infesting_plants.v1i.yolov7pytorch/test/images/"
        elif dataset == "geok":
            #self.image_dir = "segmentation/data/geok/test/images/"
            #self.image_dir = "./data/geok/train/images/"
            self.image_dir = "./small_data/geok/train/images/"
        else:
            raise ValueError("Invalid dataset selected.")

        assert model_architecture in ["slim", "squeeze"]

        self.model_architecture = model_architecture
        self.image_resolution = image_resolution
        self.width = width
        model_key = f"{dataset}_{model_architecture}_{image_resolution[0]}"

        if is_trans:
            model_key += "_trans"
        if is_best_fitting:
            model_key += "_opt"
        if model_architecture == "slim":
            self.model = SlimUNet(out_channels=2)
        elif dataset == "cofly" or is_trans:
            self.model = SlimSqueezeUNetCofly(out_channels=2)
        elif dataset == "geok":
            self.model = SlimSqueezeUNet(out_channels=2)

        print(f"model_key: {model_key}")
        self.model.load_state_dict(
            torch.load(
                Path(settings.PROJECT_DIR)
                / f"segmentation/training/garage/{model_key}.pt"
            )
        )
        self.fixed_image = fixed_image
        self.save_image = save_image
        self.adaptive_width = AdaptiveWidth(model_key)
        self.tensor_to_image = ImageImporter('geok').tensor_to_image
        self.random_image_index = -1

    def _yolov7_label(self, label, image_width, image_height):
        """
        Implement an image mask generation according to this:
        https://roboflow.com/formats/yolov7-pytorch-txt
        """
        #print("label: ")
        #print(label)
        # Deconstruct a row
        class_id, center_x, center_y, width, height = [
            float(x) for x in label.split(" ")
        ]

        # Get center pixel
        center_x = center_x * image_width
        center_y = center_y * image_height

        # Get border pixels
        top_border = int(center_x - (width / 2 * image_width))
        bottom_border = int(center_x + (width / 2 * image_width))
        left_border = int(center_y - (height / 2 * image_height))
        right_border = int(center_y + (height / 2 * image_height))

        # Generate pixels
        pixels = []
        for x in range(left_border, right_border):
            for y in range(top_border, bottom_border):
                pixels.append((x, y))

        return int(class_id), pixels

    def _get_single_image(self, filename):
        file_name = filename
        img_filename = file_name
        img = Image.open(self.project_path / self.image_dir / file_name)
        create_tensor = transforms.ToTensor()
        smaller = transforms.Resize(self.image_resolution)

        img = smaller(img)
        img = create_tensor(img)

        image_width = img.shape[1]
        image_height = img.shape[2]

        # Constructing the segmentation mask
        # We init the whole tensor as the background
        mask = torch.cat(
            (
                torch.ones(1, image_width, image_height),
                torch.zeros(1, image_width, image_height),
            ),
            0,
        )

        # Then, label by label, add to other classes and remove from background.
        file_name = file_name[:-3] + "txt"
        with open(
            self.project_path / self.image_dir.replace("images", "labels") / file_name
        ) as rows:
            labels = [row.rstrip() for row in rows]
            for label in labels:
                class_id, pixels = self._yolov7_label(label, image_width, image_height)
                if class_id != 1:
                    continue
                # Change values based on received pixels
                for pixel in pixels:
                    mask[0][pixel[0]][pixel[1]] = 0
                    mask[class_id][pixel[0]][pixel[1]] = 1

        img = img.to("cuda:0")
        mask = mask.to("cuda:0")
        img = img[None, :]
        mask = mask[None, :]

        return img, mask, img_filename

    def _generate_images(self, X, y, y_pred):
        if not os.path.exists("results"):
            os.mkdir("results")
        # Generate an original rgb image with predicted mask overlay.
        x_mask = torch.tensor(
            torch.mul(X.clone().detach().cpu(), 255), dtype=torch.uint8
        )
        x_mask = x_mask[0]

        # Draw predictions
        y_pred = y_pred[0]
        mask = torch.argmax(y_pred.clone().detach(), dim=0)
        weed_mask = torch.where(mask == 1, True, False)[None, :, :]
        # lettuce_mask = torch.where(mask == 2, True, False)[None, :, :]
        # mask = torch.cat((weed_mask, lettuce_mask), 0)

        image = draw_segmentation_masks(x_mask, weed_mask, colors=["red"], alpha=0.5)
        plt.imshow(image.permute(1, 2, 0))
        plt.savefig(f"results/{self.random_image_index}_pred.jpg")

        # Draw ground truth
        mask = y.clone().detach()[0]
        weed_mask = torch.where(mask[1] == 1, True, False)[None, :, :]
        # lettuce_mask = torch.where(mask[2] == 1, True, False)[None, :, :]
        # mask = torch.cat((weed_mask, lettuce_mask), 0)
        image = draw_segmentation_masks(x_mask, weed_mask, colors=["red"], alpha=0.5)
        plt.imshow(image.permute(1, 2, 0))
        plt.savefig(f"results/{self.random_image_index}_true.jpg")

    def infer(self, filename, fixed=-1):
        image, mask, filename = self._get_single_image(filename)

        print(f"Running inference on {filename}")

        # Select and set the model width
        self.model.set_width(self.width)

        # Get a prediction
        y_pred = self.model.forward(image)
        probs = y_pred.cpu().detach().numpy()
        probs = probs.squeeze(0) # remove batch dimension
        probs = probs.transpose(1,2,0) # rearrange dimensions to (256, 256, 2)
        gt = torch.argmax(mask, dim=1) # convert to class IDs
        gt = gt.squeeze(0) # remove batch dimension 
        gt = gt.cpu().numpy()
        metrics = Metricise()
        metrics.calculate_metrics(mask, y_pred, "test")
        results = metrics.report(None)

        # Generate overlayed segmentation masks (ground truth and prediction)
        if self.save_image:
            self._generate_images(image, mask, y_pred)

        return results, probs, gt, filename

class ModelSelectionEnv(gym.Env):
    def __init__(self, image_paths):
        super(ModelSelectionEnv, self).__init__()

        self.image_paths = image_paths  # Array of image features
        self.current_step = 0
        n_model_configurations = 4 # 0.25, 0.5, 0.75, 1.0
        
        # Define action and observation space
        # Assuming 'n_model_configurations' is the number of possible model choices
        self.action_space = spaces.Discrete(n_model_configurations)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3,), dtype=np.float32)  # Brightness in range [0, 1]
        self.tensor_to_image = ImageImporter('geok').tensor_to_image

    def reset(self):
        self.current_step = 0
        return self.calculate_features(self.image_paths[self.current_step])

    def step(self, action):
        image_path = self.image_paths[self.current_step]

        # Map the action to a model width
        width_mapping = {0: 0.25, 1: 0.5, 2: 0.75, 3: 1.0}
        chosen_width = width_mapping[action]

        print(f"Chosen width {chosen_width}")

        # Compute observation space
        next_state = self.calculate_features(self.image_paths[self.current_step])

        # Perform inference with the selected width
        inference_result, S = self.perform_inference(self.image_paths[self.current_step], chosen_width, self.current_step)

        # Then move to the next image
        print(f"length of images: {len(self.image_paths)}")
        done = self.current_step >= len(self.image_paths)
        print(f"done: {done}")

        # Reward calculation logic here
        reward = self.calculate_reward(action, self.current_step, S)
        self.current_step += 1
        print(f"self.current_step: {self.current_step}")
        done = self.current_step >= len(self.image_paths)
        return np.array([next_state], dtype=np.float32), reward, done, {}

    def calculate_features(self, image_path):
        img = Image.open(image_path)
        create_tensor = transforms.ToTensor()

        image_resolution=(
            256,
            256,
        )

        # Resize image
        smaller = transforms.Resize(image_resolution)

        img = smaller(img)
        img = create_tensor(img)

        image_width = img.shape[1]
        image_height = img.shape[2]

        img = img.to("cuda:0")
        img = img[None, :]


        # Convert the image from Tensor to a numpy array for processing with OpenCV
        np_image = self.tensor_to_image(img.cpu())[0]

        # Ensure the image has the correct shape (H, W, C) where C should be 3
        if np_image.ndim == 3 and np_image.shape[-1] != 3:
            # Handle incorrect number of channels, possibly convert to 3 channels
            # Example: np_image = np_image[:, :, :3] if there are more than 3 channels
            raise ValueError("Image has an incorrect number of channels.")

        np_image = (np_image * 255).astype(np.uint8)  # Scale from [0,1] to [0,255]

        # Create an instance of all feature classes with the numpy image
        spectral_features = SpectralFeatures(np_image)

        # Calculate 'Std Saturation', 'Hue Std', and 'Contrast' for the image
        _, hue_std, _ = spectral_features.compute_hue_histogram()
        contrast = spectral_features.compute_contrast()
        _, std_saturation, _, _ = spectral_features.compute_saturation()

        # Convert features to float and scale if necessary
        hue_std = float(hue_std)
        contrast = float(contrast)
        std_saturation = float(std_saturation)

        print(f"image path: {image_path}")
        print("hue_std: ")
        print(hue_std)
        print("contrast: ")
        print(contrast)
        print("std_saturation: ")
        print(std_saturation)

        return np.array([std_saturation, hue_std, contrast], dtype=np.float32)

    def perform_inference(self, image_path, width, step):
        image_path = image_path.split('/')[-1]

        print(f"image_path={image_path}")

        inference_engine = WidthAdjustableSingleImageInference(
            dataset='geok',
            image_resolution=(256, 256),
            model_architecture="squeeze",
            width=width,
            filename=image_path,
            save_image=True,
            is_trans=True,
            is_best_fitting=False,
        )
        results, probs, gt, filename = inference_engine.infer(image_path)

        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / e_x.sum(axis=-1, keepdims=True)

        probs_softmax = softmax(probs)
        # print("probs_softmax")
        # print(probs_softmax)
        # print("probs_softmax.shape: ")
        # print(probs_softmax.shape)
        # print("filename: ")
        # print(filename)
        # print("ground truth shape: ")
        # print(gt.shape)

        # save the probabilities and ground truth data to hdf5 file
        probs_gt_save(probs_softmax, gt, filename, step)
        
        # retrieve Metaseg metric
        print(step)
        S = compute_metrics_i(step)

        print(f"average S: {S}")
        
        return results, S

    def calculate_reward(self, action, step, S):
        # Implement the reward calculation
        # Example: Higher reward for more efficient model configuration
        return S
