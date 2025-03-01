import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from types import SimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics import YOLO
from utils import save_image



# Custom Dataset for RAW images and YOLO annotations
class RawDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, img_size=640):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load 16-bit RAW image (1 channel)
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path)
        image = np.array(image).astype(np.float32) / 4095.0  # Normalize to [0, 1]
        image = torch.tensor(image).unsqueeze(0)  # Shape: [1, H, W]

        # Load YOLO annotations
        annotation_path = os.path.join(self.annotation_dir, self.image_files[idx].replace(".png", ".txt"))
        with open(annotation_path, "r") as f:
            annotations = []
            count = 0
            for line in f.readlines():
                class_id, xc, yc, w, h = map(float, line.strip().split())
                annotations.append([class_id, xc, yc, w, h])
                count += 1
            while count < 16:
                annotations.append([-1, -1, -1, -1, -1])  # padding
                count += 1

        return image, torch.tensor(annotations)


class RawImageDataset(Dataset):
    def __init__(self, raw_dir, transform=None):
        self.raw_dir = raw_dir
        self.image_files = [f for f in os.listdir(raw_dir) if f.endswith(".png")]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.raw_dir, self.image_files[idx])
        image = Image.open(img_path)
        image = np.array(image).astype(np.float32) / 4095.0  # Normalize to [0, 1]
        image = torch.tensor(image).unsqueeze(0)  # Add channel dimension
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder with dynamic padding
        self.dec1 = self.upconv_block(1024, 512)
        self.dec2 = self.upconv_block(1024, 256)
        self.dec3 = self.upconv_block(512, 128)
        self.dec4 = self.upconv_block(256, 64)

        # Final output layer
        self.final_layer = nn.Sequential(
            nn.Upsample(size=(640, 640), mode='bilinear', align_corners=True),
            nn.Conv2d(128, 3, kernel_size=1),
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def upconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # [B, 64, 200, 300]
        e2 = self.enc2(e1)  # [B, 128, 100, 150]
        e3 = self.enc3(e2)  # [B, 256, 50, 75]
        e4 = self.enc4(e3)  # [B, 512, 25, 37] (original 400x600 input)

        # Bottleneck
        b = self.bottleneck(e4)  # [B, 1024, 12, 18]

        # Decoder with dynamic padding
        d1 = self.dec1(b)  # [B, 512, 24, 36]
        d1 = self.pad_to_match(d1, e4)
        d1 = torch.cat([d1, e4], dim=1)  # [B, 1024, 25, 37]

        d2 = self.dec2(d1)  # [B, 256, 50, 74]
        # d2 = self.pad_to_match(d2, e3)
        d2 = torch.cat([d2, e3], dim=1)  # [B, 512, 50, 75]

        d3 = self.dec3(d2)  # [B, 128, 100, 150]
        d3 = torch.cat([d3, e2], dim=1)  # [B, 256, 100, 150]

        d4 = self.dec4(d3)  # [B, 64, 200, 300]
        d4 = torch.cat([d4, e1], dim=1)  # [B, 128, 200, 300]

        return self.final_layer(d4)  # [B, 3, 640, 640]

    def pad_to_match(self, source, target):
        """Dynamically pad source tensor to match target tensor dimensions"""
        diff_h = target.size()[2] - source.size()[2]
        diff_w = target.size()[3] - source.size()[3]

        return F.pad(source,
                     (diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2))


class UNetLite(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (Smaller channels)
        self.enc1 = self.conv_block(1, 32, dropout=0.1)
        self.enc2 = self.conv_block(32, 64, dropout=0.1)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck with depthwise separable conv
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            DepthwiseSeparableConv(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Decoder with reduced channels
        self.dec1 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(512, 128)
        self.dec3 = self.upconv_block(256, 64)
        self.dec4 = self.upconv_block(128, 32)

        # Final output with efficient upsampling
        self.final_layer = nn.Sequential(
            nn.Upsample(size=(640, 640), mode='bilinear', align_corners=True),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )

    def conv_block(self, in_c, out_c, dropout=None):
        layers = [
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ]
        if dropout:
            layers.insert(2, nn.Dropout2d(dropout))
        return nn.Sequential(*layers)

    def upconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DepthwiseSeparableConv(in_c, out_c),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # [B, 32, 200, 300]
        e2 = self.enc2(e1)  # [B, 64, 100, 150]
        e3 = self.enc3(e2)  # [B, 128, 50, 75]
        e4 = self.enc4(e3)  # [B, 256, 25, 37]

        # Bottleneck
        b = self.bottleneck(e4)  # [B, 512, 25, 37]

        # Decoder with skip connections
        d1 = self.dec1(b)
        d1 = self.pad_to_match(d1, e4)
        d1 = torch.cat([d1, e4], dim=1)  # [B, 512, 25, 37]

        d2 = self.dec2(d1)
        d2 = self.pad_to_match(d2, e3)
        d2 = torch.cat([d2, e3], dim=1)  # [B, 256, 50, 75]

        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e2], dim=1)  # [B, 128, 100, 150]

        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e1], dim=1)  # [B, 64, 200, 300]

        return self.final_layer(d4)

    def pad_to_match(self, source, target):
        """Dynamic padding helper"""
        diff_h = target.size()[2] - source.size()[2]
        diff_w = target.size()[3] - source.size()[3]
        return F.pad(source, (diff_w // 2, diff_w - diff_w // 2,
                              diff_h // 2, diff_h - diff_h // 2))


class DepthwiseSeparableConv(nn.Module):
    """Lightweight convolution block"""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, kernel_size=3,
                                   padding=1, groups=in_c)
        self.pointwise = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        self.unet = UNetLite().to(device)

        # Initialize YOLO DetectionModel

        # Uncomment for train (comment out for test)
        # self.pretrained_yolo = YOLO("yolo11n.pt").to(device)
        # self.pretrained_yolo.train(data="processed_dataset.yaml", epochs=1, freeze=10)

        # Uncomment for test
        self.pretrained_yolo = YOLO("yolo11n.yaml").to(device)

        self.yolo = self.pretrained_yolo.model.to(device)
        # Manually set args for the DetectionModel
        self.yolo.args = SimpleNamespace(
            data="processed_dataset.yaml",
            nc=3,
            reg_max=16,  # Default value for YOLOv8
            iou=0.7,  # IoU threshold
            box=7.5,  # Box loss gain
            cls=0.5,  # Classification threshold
            dfl=1.5,  # DFL loss gain
            cls_pw=1.0,  # Classification positive weight
            obj_pw=1.0,  # Objectness positive weight
            fl_gamma=0.0  # Focal loss gamma
        )

        # Initialize the loss function
        self.yolo.loss = v8DetectionLoss(self.yolo)
        self.count = 0

    def forward(self, x):
        # Process through UNet
        unet_out = self.unet(x)

        # for saving intermediate images
        # save_image("intermediate_imgs", unet_out, 0, self.count)
        # self.count += 1

        # Get YOLO outputs
        return self.yolo(unet_out)

def xywh_to_xyxy(boxes):
    """Convert center coordinates (x, y, w, h) to (x1, y1, x2, y2)"""
    x, y, w, h = boxes.unbind(-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


