# YOLO v2 From Scratch

This repository contains the implementation of YOLO v2 (You Only Look Once version 2) from scratch using PyTorch.

## Introduction

YOLO v2 is a popular real-time object detection algorithm that is known for its speed and accuracy. This implementation aims to provide a clear and understandable codebase for training and testing YOLO v2 on custom datasets.

## Files

- `darknet_dataset.py`: Contains the dataset class for preparing data for training the Darknet.
- `darknet_testing.py`: Script for testing the Darknet net.
- `darknet_training.py`: Script for training the Darknet net.
- `utils.py`: Utility functions used across the project.
- `yolo_dataset.py`: Contains the dataset class for preparing data for training and testing in YOLO format.
- `yolo_loss.py`: Implementation of the YOLO loss function.
- `yolo_results.py`: Script for analyzing and visualizing the results of YOLO v2.
- `yolo_testing.py`: Script for testing YOLO v2 on images using YOLO format.
- `yolo_training.py`: Script for training YOLO v2 on custom datasets in YOLO format.

## Usage

1. **Data Preparation**: Prepare your dataset.
2. **Darknet training**: Use `darknet_training.py` to train the Darknet on your dataset.
3. **Darknet testing**: Use `darknet_testing.py` to test the trained model on the test set.
4. **YOLOv2 training**: Use `yolo_training.py` to train the YoloV2 model on your dataset.
5. **Results Analysis**: Use `yolo_results.py` to analyze and visualize the results of the YoloV2 model.
6. **Testing**: Use `yolo_testing.py` to compute the mAP of the model.

## References

- Original YOLO v2 paper: [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)


