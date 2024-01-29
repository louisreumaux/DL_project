import torch
from net.yolo_net import YOLO
from utils import output_tensor_to_boxes, nonmax_suppression, visualize_bbox, plot_img, target_tensor_to_boxes
import pickle
from yolo_dataset import CarDataset
from torch.utils.data import DataLoader
import json


''' 
This script is used to visualise the results of a YOLOv2 model trained on a custom dataset with given weights. 
'''


torch.manual_seed(1)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

with open('parameters/yolo_parameters.json', 'r') as f:
    parameters = json.load(f)


anchors = parameters["anchors"]
num_classes = parameters["classes"]
test_images_folder = "data_yolo/testing_images/"

model = YOLO(num_classes, anchors)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)

# Load the weights from the checkpoint file
weight_folder = "net/yolo_weights/"
checkpoint_path = 'yolo_model_v2.pth'
checkpoint = torch.load(weight_folder+checkpoint_path)
model.load_state_dict(checkpoint)

model.eval()


with open('data_yolo/dictionnary_bounding_boxes_test.pkl', 'rb') as fp:
    data_bounding_boxes = pickle.load(fp)


test_dataset = CarDataset(data=data_bounding_boxes, input_size=416, images_folder=test_images_folder, data_augmentation = False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, drop_last=True)

imgs, targets = next(iter(test_loader))

targets = targets.to(device)


output = model(imgs.to(device))

for i in range(len(imgs)):

    output_tensor = output[i].detach().cpu()
    output_tensor = output_tensor.permute(1,2,0)
    output_tensor = output_tensor.view(13, 13, 5, 5)
    boxes = output_tensor_to_boxes(output_tensor, anchors)
    boxes = nonmax_suppression(boxes,0.2)
    img = imgs[i].permute(1,2,0).cpu().numpy()

    true_boxes = target_tensor_to_boxes(targets[i])
    img = visualize_bbox(img.copy(),color=(255,0,0), boxes=boxes, thickness=1, draw_center=False)
    img = visualize_bbox(img.copy(),color=(0,255,0), boxes=true_boxes, thickness=1, draw_center=False)
    plot_img(img, size=(4,4))