import os
from pathlib import Path
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
from segmentation.data.data import ImageImporter


class WidthAdjustableSingleImageInference:
    def __init__(
        self,
        dataset,
        image_resolution,
        model_architecture,
        width,
        filename,
        train,
        fixed_image=-1,
        save_image=False,
        is_trans=False,
        is_best_fitting=False,
    ):
        self.project_path = Path(settings.PROJECT_DIR)
        if dataset == "infest":
            self.image_dir = "segmentation/data/agriadapt/NN_labeled_samples_salad_infesting_plants.v1i.yolov7pytorch/test/images/"
        elif dataset == "geok":
            if train:
                self.image_dir = "./data/geok/train/images/"
            else:
                self.image_dir = "./data/geok/test/images/"

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