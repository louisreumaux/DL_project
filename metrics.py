from utils import target_tensor_to_boxes, output_tensor_to_boxes, nonmax_suppression, compute_output_iou, ap_computation
from yolo_dataset import CarDataset
from net.yolo_net import YOLO
import torch
import pickle
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score




def plot_width_height_graph(bboxes_wh, bboxes_wh_well_predicted):
    bboxes_wh = np.array(bboxes_wh)
    bboxes_wh_well_predicted = np.array(bboxes_wh_well_predicted)

    x = np.linspace(0, 200, 400)
    y = np.linspace(0, 160, 400)

    X, Y = np.meshgrid(x, y)

    Z = X * Y

    mask1 = Z < 32**2
    mask2 =  (32**2 <= Z) & (Z < 96**2)
    mask3 = 96**2 <= Z 

    # Créer le graphique
    plt.figure(figsize=(8, 8))

    # Tracer la zone colorée avec le masque
    plt.pcolormesh(X, Y, mask1, cmap='viridis', shading='auto', alpha=0.5)
    plt.pcolormesh(X, Y, mask2, cmap='viridis', shading='auto', alpha=0.5)
    plt.pcolormesh(X, Y, mask3, cmap='viridis', shading='auto', alpha=0.5)


    plt.scatter(bboxes_wh[:, 0], bboxes_wh[:, 1], alpha=0.5, edgecolors='w', color='red')
    plt.scatter(bboxes_wh_well_predicted[:, 0], bboxes_wh_well_predicted[:, 1], linewidths=1,edgecolors='w', color='green')
    plt.title('Localisation of well predicted boxes')
    plt.xlabel('Width')
    plt.ylabel('Height')

    plt.show()

def mean_avg_precision(model,device,test_loader, iou_threshold):
    predictions = []
    labels = []
    wh_boxes_well_predicted = []
    wh_boxes = []
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        for i in range(len(outputs)):
            output_tensor = outputs[i].detach().cpu()
            output_tensor = output_tensor.permute(1,2,0)
            output_tensor = output_tensor.view(13, 13, 5, 5)

            target_tensor = targets[i].detach().cpu()

            pred_boxes = output_tensor_to_boxes(output_tensor)
            pred_boxes = nonmax_suppression(pred_boxes,0.2)

            true_boxes = target_tensor_to_boxes(target_tensor)

            pred_y, true_y, wh_boxes_well_predicted_i, wh_boxes_i = compute_output_iou(pred_boxes, true_boxes, iou_threshold)

            predictions += pred_y
            labels += true_y
            wh_boxes_well_predicted += wh_boxes_well_predicted_i
            wh_boxes += wh_boxes_i

    AP = ap_computation(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    return AP, precision, recall, wh_boxes_well_predicted, wh_boxes


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

AP, precision, recall, bboxes_wh_well_predicted, bboxes_wh = mean_avg_precision(model, device, train_loader, 0.25)
print(f"Training mAP 0.25 : {AP:.4f}, Recall : {recall:.4f}, Precision : {precision:.4f}")
plot_width_height_graph(bboxes_wh, bboxes_wh_well_predicted)

AP, precision, recall, bboxes_wh_well_predicted, bboxes_wh = mean_avg_precision(model, device, train_loader, 0.5)
print(f"Training mAP 0.5 : {AP:.4f}, Recall : {recall:.4f}, Precision : {precision:.4f}")
plot_width_height_graph(bboxes_wh, bboxes_wh_well_predicted)

AP, precision, recall, bboxes_wh_well_predicted, bboxes_wh = mean_avg_precision(model, device, train_loader, 0.75)
print(f"Training mAP 0.75 : {AP:.4f}, Recall : {recall:.4f}, Precision : {precision:.4f}")
plot_width_height_graph(bboxes_wh, bboxes_wh_well_predicted)



### TEST SET RESULTS ###

with open('data_yolo/dictionnary_bounding_boxes_test.pkl', 'rb') as fp:
    data_bounding_boxes = pickle.load(fp)

test_dataset = CarDataset(data=data_bounding_boxes, input_size=416, images_folder=test_images_folder, geometrical_modifications = False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, drop_last=True)

AP, precision, recall, bboxes_wh_well_predicted, bboxes_wh = mean_avg_precision(model, device, test_loader, 0.25)
print(f"Testing mAP 0.25 : {AP:.4f}, Recall : {recall:.4f}, Precision : {precision:.4f}")
plot_width_height_graph(bboxes_wh, bboxes_wh_well_predicted)


AP, precision, recall, bboxes_wh_well_predicted, bboxes_wh = mean_avg_precision(model, device, test_loader, 0.5)
print(f"Testing mAP 0.5 : {AP:.4f}, Recall : {recall:.4f}, Precision : {precision:.4f}")
plot_width_height_graph(bboxes_wh, bboxes_wh_well_predicted)


AP, precision, recall, bboxes_wh_well_predicted, bboxes_wh = mean_avg_precision(model, device, test_loader, 0.75)
print(f"Testing mAP 0.75 : {AP:.4f}, Recall : {recall:.4f}, Precision : {precision:.4f}")
plot_width_height_graph(bboxes_wh, bboxes_wh_well_predicted)






