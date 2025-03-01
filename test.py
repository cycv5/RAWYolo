import random
import torch
import torchvision
import torchvision.transforms as T
import torchvision.utils as utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader, Dataset
from model import RawDataset, UNet, CombinedModel
import time

def decode_yolov8_outputs(raw_outputs, conf_threshold=0.52, iou_threshold=0.45):
    """
    Decodes raw YOLOv8 outputs into a list of detection dictionaries.
    """
    # Use only the first element which contains detection predictions
    det = raw_outputs[0]  # shape: (B, 84, 8400)

    # Transpose to (B, num_boxes, 84)
    pred = det.permute(0, 2, 1)  # shape: (B, 8400, 84)
    batch_size = pred.shape[0]
    num_boxes = pred.shape[1]
    num_classes = pred.shape[2] - 4  # 4 for [x, y, w, h]

    # Separate predictions: first 4 channels are boxes, next 80 are class scores.
    boxes_xywh = pred[..., :4]  # (B, num_boxes, 4)
    class_logits = pred[..., 4:]  # (B, num_boxes, num_classes)


    # Apply sigmoid to the box center coordinates (if they are normalized) and class scores.
    # Depending on your training, you might also need to apply a sigmoid to the box coordinates.
    boxes_xywh = boxes_xywh / 640.0
    class_scores = torch.sigmoid(class_logits)

    # print(boxes_xywh)
    # print(class_scores)

    # For YOLOv8, the objectness score is typically merged into the class scores.
    # For each detection, choose the best class.
    cls_conf, cls_pred = class_scores.max(dim=-1)  # (B, num_boxes)

    # Convert from center (x, y, w, h) to corner (x1, y1, x2, y2)
    x = boxes_xywh[..., 0]
    y = boxes_xywh[..., 1]
    w = boxes_xywh[..., 2]
    h = boxes_xywh[..., 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, num_boxes, 4)

    outputs = []
    for i in range(batch_size):
        # Filter out detections with low confidence
        valid_mask = cls_conf[i] > conf_threshold
        if valid_mask.sum() == 0:
            outputs.append({
                "boxes": torch.empty((0, 4), device=det.device),
                "scores": torch.empty((0,), device=det.device),
                "labels": torch.empty((0,), dtype=torch.int64, device=det.device)
            })
            continue

        boxes_i = boxes[i][valid_mask]
        scores_i = cls_conf[i][valid_mask]
        labels_i = cls_pred[i][valid_mask]

        # Apply Non-Maximum Suppression (NMS)
        keep = torchvision.ops.nms(boxes_i, scores_i, iou_threshold)
        outputs.append({
            "boxes": boxes_i[keep],
            "scores": scores_i[keep],
            "labels": labels_i[keep]
        })

    return outputs


def decode_targets(targets):
    """
    Converts targets with shape (batch_size, 16, 5) into a list of target dictionaries.
    """
    batch = targets.shape[0]
    targs = []
    for i in range(batch):
        # Select rows where the class is not -1 (i.e. valid objects)
        valid = targets[i][:, 0] != -1
        valid_targets = targets[i][valid]
        if valid_targets.shape[0] == 0:
            boxes = torch.empty((0, 4))
            labels = torch.empty((0,), dtype=torch.int64)
        else:
            labels = valid_targets[:, 0].to(torch.int64)
            x, y, w, h = valid_targets[:, 1], valid_targets[:, 2], valid_targets[:, 3], valid_targets[:, 4]
            # Convert from center format to corner format: (x1, y1, x2, y2)
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
        targs.append({"boxes": boxes, "labels": labels})
    return targs


def show_img(image_tensor, detections):
    label_to_txt = {0: "car", 1: "person", 2: "bicycle"}
    image_tensor = (image_tensor * 255).to(torch.uint8)
    # Convert grayscale to RGB by repeating the single channel 3 times
    image_tensor = image_tensor.expand(3, -1, -1)

    # Convert normalized box coordinates to absolute pixel coordinates
    boxes = detections['boxes'].clone()
    boxes[:, [0, 2]] *= image_tensor.shape[2]  # Scale x coordinates by image width
    boxes[:, [1, 3]] *= image_tensor.shape[1]  # Scale y coordinates by image height

    # Draw bounding boxes on the image
    image_with_boxes = utils.draw_bounding_boxes(
        image_tensor,
        boxes=boxes,
        labels=[label_to_txt[label.item()] for label in detections['labels']],
        colors="red",
        width=2
    )

    # Convert the tensor to a PIL image and display
    plt.figure(figsize=(10, 8))
    plt.imshow(T.ToPILImage()(image_with_boxes))
    plt.axis('off')  # Hide axes
    plt.show()


def evaluate(model, test_loader, device='cuda'):
    """
    Evaluates the model against the ground truth targets provided by test_loader.
    """
    # Create the metric object with IoU thresholds ranging from 0.5 to 0.95
    metric = MeanAveragePrecision(iou_thresholds=[0.5000, 0.5500, 0.6000, 0.6500, 0.7000, 0.7500, 0.8000, 0.8500, 0.9000, 0.9500], class_metrics=True)

    all_preds = []
    all_targets = []

    model.to(device)
    model.yolo.eval()
    model.unet.eval()

    start_time = time.time()
    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = images.to(device)
            targets = targets.to(device)
            # Obtain the raw outputs from the model
            raw_outputs = model(images)  # call model directly to get raw output tensor

            # Decode raw outputs to get list of detections (one dict per image)
            preds = decode_yolov8_outputs(raw_outputs)
            # Convert ground truth targets to the required dict format
            targs = decode_targets(targets)
            all_preds.extend(preds)
            all_targets.extend(targs)
            # r = random.randint(0, 10)
            # print(images[r])
            # show_img(images[r], preds[r])
    print(f"--- Average inference time for 1 image: {((time.time() - start_time) / 427.0 * 1000)} ms ---")

    # for i in range(10):
    #     print("preds")
    #     print(all_preds[i])
    #     print("targs")
    #     print(all_targets[i])
    #     print("=============")

    # Update and compute the metric
    metric.update(all_preds, all_targets)
    results = metric.compute()
    # results is a dict with keys such as 'map50' (mAP@0.5) and 'map' (mAP@0.50:0.95)
    print("All Results:")
    print(results)
    print("=====")
    print("Key Results:")
    print("mAP@0.50:0.95:", results['map'])
    print("mAP@0.5:", results['map_50'])
    print("mAP@0.75:", results['map_75'])
    print("mAP per class (car, person, bicycle):", results['map_per_class'])


if __name__ == '__main__':
    model_test = CombinedModel()
    model_test.load_state_dict(torch.load("combined_model_30.pth"))
    test_dataset = RawDataset(image_dir="dataset_split/test/images", annotation_dir="dataset_split/test/labels")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    evaluate(model_test, test_loader)
