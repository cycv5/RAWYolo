import os
import xml.etree.ElementTree as ET

# Path to your annotations directory
annotations_dir = "dataset/Annotations"  # Folder containing XML files

# Class mapping (replace with your dataset's classes)
class_mapping = {
    "car": 0,
    "person": 1,
    "bicycle": 2,
    # Add more classes as needed
}

# Function to parse PASCAL VOC XML annotations
def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Get image dimensions
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

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

    return width, height, boxes, labels

# Function to convert bounding boxes to YOLO format
def convert_to_yolo_format(box, width, height):
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2 / width  # Normalized x_center
    y_center = (ymin + ymax) / 2 / height  # Normalized y_center
    box_width = (xmax - xmin) / width  # Normalized width
    box_height = (ymax - ymin) / height  # Normalized height
    return x_center, y_center, box_width, box_height

# Process all XML files in the annotations directory
for xml_file in os.listdir(annotations_dir):
    if not xml_file.endswith(".xml"):
        continue

    # Parse XML file
    xml_path = os.path.join(annotations_dir, xml_file)
    width, height, boxes, labels = parse_annotation(xml_path)

    # Prepare YOLO-style annotations
    yolo_annotations = []
    for box, label in zip(boxes, labels):
        if label not in class_mapping:
            continue  # Skip unknown classes
        class_id = class_mapping[label]
        x_center, y_center, box_width, box_height = convert_to_yolo_format(box, width, height)
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    # Save YOLO-style annotations to TXT file (same directory)
    txt_file = os.path.splitext(xml_file)[0] + ".txt"
    txt_path = os.path.join(annotations_dir, txt_file)
    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_annotations))

    print(f"Converted {xml_file} to {txt_file}")

print("Conversion complete!")