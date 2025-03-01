# YOLO on RAW - NO Need for ISP
A UNet+YOLO network for object detection on RAW images.

The code is in test mode, and the trained weight is combined_model_30.pth

To run the code, install the libraries needed and do
```
python test.py
```
Libraries needed include:
torch, torchvision, matplotlib.pyplot, tqdm, torchmetrics, ultralytics, numpy

Note that data is organized in the dataset_split folder, only 16 test images and labels are included here, but more can be added. Training images and labels can be placed with the same organization under dataset_split/train/. The labels are in the YOLO format, which is converted from PASCAL VOC XML using convert_annotation.py
