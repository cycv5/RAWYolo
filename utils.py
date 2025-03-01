import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image
import random
import shutil
import torch

# Paths to your images and annotations
image_path = "dataset/Images/2014_001000.png"  # Replace with your RAW image path
annotation_path = "dataset/Annotations/2014_001000.xml"  # Replace with corresponding annotation path

# Function to parse PASCAL VOC XML annotations
def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    return boxes, labels

# Function to load 16-bit PNG image
def load_16bit_png(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)  # Convert to numpy array
    return image_array

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, labels):
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Draw rectangle
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Add label

def check_bboxes(image, boxes, labels):
    # Load 16-bit PNG image
    image_array = load_16bit_png(image_path)
    # Parse annotations
    boxes, labels = parse_annotation(annotation_path)
    # Normalize 16-bit image to 8-bit for visualization
    image_normalized = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Draw bounding boxes on the normalized image
    draw_boxes(image_normalized, boxes, labels)
    # Display the image with bounding boxes
    plt.imshow(image_normalized, cmap="gray")
    plt.title("16-bit RAW Image with Annotations")
    plt.axis("off")
    plt.show()


def split_dataset(images_dir, annotations_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Splits the dataset into train, validation, and test sets.

    Args:
        images_dir (str): Path to the folder containing raw images.
        annotations_dir (str): Path to the folder containing annotations (TXT and XML).
        output_dir (str): Path to the output directory where the split dataset will be saved.
        train_ratio (float): Proportion of the dataset for training.
        val_ratio (float): Proportion of the dataset for validation.
        test_ratio (float): Proportion of the dataset for testing.
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]
    random.seed(42)
    random.shuffle(image_files)  # Shuffle for random splitting

    # Split indices
    num_images = len(image_files)
    train_end = int(train_ratio * num_images)
    val_end = train_end + int(val_ratio * num_images)

    # Copy files to respective folders
    for i, img_file in enumerate(image_files):
        base_name = os.path.splitext(img_file)[0]

        # Determine split
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"

        # Copy image
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(output_dir, split, "images", img_file)
        shutil.copy(src_img, dst_img)

        # Copy annotation (TXT)
        src_txt = os.path.join(annotations_dir, f"{base_name}.txt")
        dst_txt = os.path.join(output_dir, split, "labels", f"{base_name}.txt")
        shutil.copy(src_txt, dst_txt)

        # Copy annotation (XML) if needed
        src_xml = os.path.join(annotations_dir, f"{base_name}.xml")
        if os.path.exists(src_xml):
            dst_xml = os.path.join(output_dir, split, "labels", f"{base_name}.xml")
            shutil.copy(src_xml, dst_xml)

    print(f"Dataset split into train ({train_ratio * 100}%), val ({val_ratio * 100}%), test ({test_ratio * 100}%)")


def prepare_batch(raw_images, targets):
    """
    Prepares the batch dictionary for the v8DetectionLoss function.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, max_objects, _ = targets.shape

    # Filter out entries where all values are -1 (indicating no object)
    mask = targets.sum(dim=2) != -5

    # Extract class IDs
    cls = [targets[i, mask[i], 0] for i in range(batch_size)]
    cls = torch.cat(cls).long()

    # Extract bounding boxes
    bboxes = [targets[i, mask[i], 1:] for i in range(batch_size)]
    bboxes = torch.cat(bboxes).float()

    # Create batch_idx tensor
    batch_idx = [torch.full((mask[i].sum().item(),), i) for i in range(batch_size)]
    batch_idx = torch.cat(batch_idx).long()

    # Create the batch dictionary
    batch = {
        'img': raw_images.to(device),
        'cls': cls.to(device),
        'bboxes': bboxes.to(device),
        'batch_idx': batch_idx.to(device)
    }

    return batch

def save_image(save_dir, batch, i, id=""):
    # Select image at index i
    os.makedirs(save_dir, exist_ok=True)
    img = batch[i].cpu().numpy()  # Convert to NumPy array
    img = img.transpose(1, 2, 0)  # Convert from [C, H, W] to [H, W, C]

    # Save image
    save_path = os.path.join(save_dir, f"image_{i}_{id}.png")
    plt.imsave(save_path, img)

    print(f"Image {i} saved at {save_path}")

if __name__ == "__main__":
    # check_bboxes(image_path, parse_annotation(annotation_path), parse_annotation(annotation_path))

    # split_dataset(
    #     images_dir="dataset/Images",
    #     annotations_dir="dataset/Annotations",
    #     output_dir="dataset_split",
    #     train_ratio=0.8,
    #     val_ratio=0.1,
    #     test_ratio=0.1
    # )
    pass

