import os
import torch
import numpy as np
from time import time
from PIL import Image
import csv
from datetime import datetime
from torchvision import transforms
from tqdm import tqdm
from shapely import Polygon, Point
from segmentation.helpers.metricise import Metricise
from numpy import floor, ceil

class RandomSegmentExperiment:
    def __init__(self, image_dir, label_dir, resolution=(512, 512), device='cuda'):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.resolution = resolution
        self.device = torch.device(device)
        self.image_list = sorted(os.listdir(image_dir))

    def _yolov7_label(self, label, image_width, image_height):
        label = label.split(" ")
        if label[0] == "0":
            return None, None
        
        if len(label) == 5:
            class_id, center_x, center_y, width, height = [float(x) for x in label]
            center_x = center_x * image_width
            center_y = center_y * image_height
            top_border = int(center_x - (width / 2 * image_width))
            bottom_border = int(center_x + (width / 2 * image_width))
            left_border = int(center_y - (height / 2 * image_height))
            right_border = int(center_y + (height / 2 * image_height))
            pixels = []
            for x in range(left_border, right_border):
                for y in range(top_border, bottom_border):
                    pixels.append((x, y))
        else:
            class_id = label[0]
            points = [(float(label[i]) * image_width, float(label[i + 1]) * image_height)
                     for i in range(1, len(label), 2)]
            poly = Polygon(points)
            pixels = []
            for x in range(int(floor(min([x[1] for x in points]))),
                         int(ceil(max([x[1] for x in points])))):
                for y in range(int(floor(min([x[0] for x in points]))),
                             int(ceil(max([x[0] for x in points])))):
                    if Point(y, x).within(poly):
                        pixels.append((x, y))
        return int(class_id), pixels

    def _get_single_image(self, file_name):
        img = Image.open(os.path.join(self.image_dir, file_name)).convert('RGB')
        create_tensor = transforms.ToTensor()
        smaller = transforms.Resize(self.resolution)
        img = smaller(img)
        img = create_tensor(img)

        image_width = img.shape[1]
        image_height = img.shape[2]

        mask = torch.cat((
            torch.ones(1, image_width, image_height),
            torch.zeros(1, image_width, image_height),
        ), 0)

        # Process ground truth labels
        label_file = file_name[:-3] + "txt"
        label_path = os.path.join(self.label_dir, label_file)
        if os.path.exists(label_path):
            with open(label_path) as rows:
                labels = [row.rstrip() for row in rows]
                for label in labels:
                    class_id, pixels = self._yolov7_label(label, image_width, image_height)
                    if class_id != 1:
                        continue
                    for pixel in pixels:
                        mask[0][pixel[0]][pixel[1]] = 0
                        mask[class_id][pixel[0]][pixel[1]] = 1

        img = img.to(self.device)
        mask = mask.to(self.device)
        img = img[None, :]
        mask = mask[None, :]
        
        return img, mask

    def infer(self, file_name):
        image, mask = self._get_single_image(file_name)
        
        # Generate random predictions
        random_pred = torch.rand(mask.shape, device=self.device)
        random_pred = (random_pred > 0.5).float()
        
        # Calculate metrics
        metrics = Metricise()
        metrics.calculate_metrics(mask, random_pred, "test")
        results = metrics.report(None)
        
        return results

    def run_experiment(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f'random_segmentation_{timestamp}.csv'
        csv_path = os.path.join('./results/random_segmentation/', csv_filename)

        os.makedirs('./results/random_segmentation/', exist_ok=True)

        total_iou = 0
        with open(csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Filename', 'IoU'])

            for image_name in tqdm(self.image_list, desc="Processing Images"):
                results = self.infer(image_name)
                iou = results['test/iou/weeds']
                total_iou += iou
                
                csvwriter.writerow([image_name, iou])

        average_iou = total_iou / len(self.image_list)
        print(f"Experiment completed. Average IoU: {average_iou:.4f}")
        print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    image_dir = "./data/ordered_train_test/all/images"
    label_dir = "./data/ordered_train_test/all/labels"

    random_experiment = RandomSegmentExperiment(
        image_dir=image_dir,
        label_dir=label_dir
    )
    
    random_experiment.run_experiment()