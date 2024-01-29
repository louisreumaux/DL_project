from utils import target_tensor_to_boxes, output_tensor_to_boxes, nonmax_suppression, compute_output_iou, ap_computation, plot_width_height_graph, mean_avg_precision
from yolo_dataset import CarDataset
from net.yolo_net import YOLO
import torch
import pickle
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt

'''
This script evaluates the performance of a YOLOv2 model trained on a custom dataset using mean average precision (mAP) metric.
'''

with open('parameters/yolo_parameters.json', 'r') as f:
    parameters = json.load(f)

anchors = parameters["anchors"]
num_classes = parameters["classes"]
training_images_folder = "data_yolo/training_images/"
test_images_folder = "data_yolo/testing_images/"

model = YOLO(num_classes, anchors)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()

# Load the weights from the checkpoint file
weight_folder = "net/yolo_weights/"
checkpoint_path = 'yolo_model_v2.pth'
checkpoint = torch.load(weight_folder+checkpoint_path)
model.load_state_dict(checkpoint)


### TRAINING SET RESULTS ###

with open('data_yolo/dictionnary_bounding_boxes.pkl', 'rb') as fp:
    data_bounding_boxes = pickle.load(fp)


train_dataset = CarDataset(data=data_bounding_boxes, input_size=416, images_folder=training_images_folder, data_augmentation = False)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

AP, precision, recall, bboxes_wh_well_predicted, bboxes_wh = mean_avg_precision(model, device, train_loader, 0.25, anchors)
print(f"Training mAP 0.25 : {AP:.4f}, Recall : {recall:.4f}, Precision : {precision:.4f}")
plot_width_height_graph(bboxes_wh, bboxes_wh_well_predicted)

AP, precision, recall, bboxes_wh_well_predicted, bboxes_wh = mean_avg_precision(model, device, train_loader, 0.5, anchors)
print(f"Training mAP 0.5 : {AP:.4f}, Recall : {recall:.4f}, Precision : {precision:.4f}")
plot_width_height_graph(bboxes_wh, bboxes_wh_well_predicted)

AP, precision, recall, bboxes_wh_well_predicted, bboxes_wh = mean_avg_precision(model, device, train_loader, 0.75, anchors)
print(f"Training mAP 0.75 : {AP:.4f}, Recall : {recall:.4f}, Precision : {precision:.4f}")
plot_width_height_graph(bboxes_wh, bboxes_wh_well_predicted)



### TEST SET RESULTS ###

with open('data_yolo/dictionnary_bounding_boxes_test.pkl', 'rb') as fp:
    data_bounding_boxes = pickle.load(fp)

test_dataset = CarDataset(data=data_bounding_boxes, input_size=416, images_folder=test_images_folder, geometrical_modifications = False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, drop_last=True)

AP, precision, recall, bboxes_wh_well_predicted, bboxes_wh = mean_avg_precision(model, device, test_loader, 0.25, anchors)
print(f"Testing mAP 0.25 : {AP:.4f}, Recall : {recall:.4f}, Precision : {precision:.4f}")
plot_width_height_graph(bboxes_wh, bboxes_wh_well_predicted)


AP, precision, recall, bboxes_wh_well_predicted, bboxes_wh = mean_avg_precision(model, device, test_loader, 0.5, anchors)
print(f"Testing mAP 0.5 : {AP:.4f}, Recall : {recall:.4f}, Precision : {precision:.4f}")
plot_width_height_graph(bboxes_wh, bboxes_wh_well_predicted)


AP, precision, recall, bboxes_wh_well_predicted, bboxes_wh = mean_avg_precision(model, device, test_loader, 0.75, anchors)
print(f"Testing mAP 0.75 : {AP:.4f}, Recall : {recall:.4f}, Precision : {precision:.4f}")
plot_width_height_graph(bboxes_wh, bboxes_wh_well_predicted)






