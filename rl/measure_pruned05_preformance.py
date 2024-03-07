from PIL import Image
import os
import torch
import pandas as pd
from torchvision import transforms
from segmentation.models.unet import unet
from segmentation.data.data import ImageImporter
from segmentation.helpers.metricise import Metricise

image_resolution = (512,512)
model_architecture = f"unet_{image_resolution[0]}_pruned"
width = "05"
model_key = f"{model_architecture}_{width}_iterative_1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = unet(n_channels=3, n_classes=2, bilinear=False)
model = model.to(device)
model.load_state_dict(
            torch.load(
                f"./training/garage/image_resolution0{image_resolution[0]}/{model_key}.pt",
                map_location=device
            )
        )
model.eval()
dataset = "geok"
tensor_to_image = ImageImporter(dataset).tensor_to_image
image_dir = './all/images'
label_dir = './all/labels'

def _yolov7_label(label, image_width, image_height):
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

def _get_single_image(file_name, image_dir, label_dir):
    img_filename = file_name
    img = Image.open(os.path.join(image_dir, file_name)).convert('RGB')
    create_tensor = transforms.ToTensor()
    smaller = transforms.Resize(image_resolution)

    img = smaller(img)
    img = create_tensor(img)

    image_width = img.shape[1]
    image_height = img.shape[2]

    one_time = True
    if one_time:
        print("image_width")
        print(image_width)

        print("image_height")
        print(image_height)
        one_time = False

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
    label_file = file_name[:-3] + "txt"
    label_path = os.path.join(label_dir, label_file)
    if os.path.exists(label_path):
        with open(label_path) as rows:
            labels = [row.rstrip() for row in rows]
            for label in labels:
                class_id, pixels = _yolov7_label(label, image_width, image_height)
                if class_id != 1:
                    continue
                # Change values based on received pixels
                for pixel in pixels:
                    mask[0][pixel[0]][pixel[1]] = 0
                    mask[class_id][pixel[0]][pixel[1]] = 1

    img = img.to(device)
    mask = mask.to(device)
    img = img[None, :]
    mask = mask[None, :]

    return img, mask, img_filename

def infer05(file_name):
    image, mask, filename = _get_single_image(file_name, image_dir, label_dir)
    
    y_pred = model(image)
    probs = y_pred.cpu().detach().numpy()
    probs = probs.squeeze(0) # remove batch dimension
    probs = probs.transpose(1,2,0) # rearrange dimensions to (256, 256, 2)
    gt = torch.argmax(mask, dim=1) # convert to class IDs
    gt = gt.squeeze(0) # remove batch dimension 
    gt = gt.cpu().numpy()
    metrics = Metricise()
    metrics.calculate_metrics(mask, y_pred, "test")
    results = metrics.report(None)

    # TODO
    # Generate overlayed segmentation masks (ground truth and prediction)
    #if self.save_image:
    #    self._generate_images(image, mask, y_pred)

    return results, probs, gt, filename


if __name__ == "__main__":
    
    all_images_path = "./all/images/"
    csv_file_path = './features.csv'
    df = pd.read_csv(csv_file_path)

    for path in os.listdir(all_images_path):
        results, probs, gt, filename = infer05(path)

        df.loc[df['Filename'] == filename, 'pruned_05_performance'] = results['test/iou/weeds']

        print(f"Filename: {filename}")
        print("Results: ")
        print(results)

    df.to_csv('./features.csv', index=False)