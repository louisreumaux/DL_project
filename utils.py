import matplotlib.pyplot as plt
import torch
import cv2
import json
import numpy as np
from shapely.geometry import Polygon
import pickle
import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score

with open('parameters/yolo_parameters.json', 'r') as f:
    parameters = json.load(f)

anchors = parameters["anchors"]

nb_anchors = len(anchors)
nb_classes = parameters["classes"]
input_size = parameters["input_size"]
grid_size = parameters["grid_size"]


def plot_img(img, size=(7,7)):
    plt.figure(figsize=size)
    plt.imshow(img[:,:,::-1])
    plt.show()
    
def visualize_bbox(img, boxes, thickness=2, color=(0, 0, 255), draw_center=True):
    '''
    Draw the rectangles described in boxes on the image img.
    '''
    img_copy = img.cpu().permute(1,2,0).numpy() if isinstance(img, torch.Tensor) else img.copy()
    for box in boxes:
        x,y,w,h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        img_copy = cv2.rectangle(
            img_copy,
            (x,y),(x+w, y+h),
            color, thickness)
        if draw_center:
            center = (x+w//2, y+h//2)
            img_copy = cv2.circle(img_copy, center=center, radius=3, color=(0,255,0), thickness=2)
    return img_copy



def target_tensor_to_boxes(boxes_tensor, output_thresh = 0.6, grid_size=grid_size, input_size=input_size):
    '''
    Convert target tensor to bounding boxes.
    
    Args:
        boxes_tensor (torch.Tensor): Target tensor containing bounding box information.
        output_thresh (float): Threshold for considering object presence.
        grid_size (int): Size of the grid used for dividing the image.
        input_size (int): Size of the input image.
        
    Returns:
        numpy.ndarray: Bounding boxes extracted from the target tensor.
    '''
    cell_w, cell_h = input_size/grid_size, input_size/grid_size
    boxes = []
    for i in range(grid_size):
        for j in range(grid_size):
            data = boxes_tensor[i,j,0]
            x_center,y_center, w, h, obj_prob = data[0], data[1], data[2], data[3], data[4]
            prob = obj_prob
            if prob > output_thresh:
                x, y = (x_center+j)*cell_w-w/2, (y_center+i)*cell_h-h/2
                box = [x,y,w,h]
                boxes.append(box)
            
    return torch.Tensor(boxes).numpy()



def boxes_to_tensor(boxes, grid_size=grid_size, nb_anchors=nb_anchors, input_size=input_size, nb_classes=nb_classes):
    '''
    Build the target tensor from bounding box coordinates.
    
    Args:
        boxes (list): List of bounding boxes.
        grid_size (int): Size of the grid used for dividing the image.
        nb_anchors (int): Number of anchors.
        input_size (int): Size of the input image.
        nb_classes (int): Number of classes.
        
    Returns:
        torch.Tensor: Target tensor containing bounding box information.
    '''
    boxes_tensor = torch.zeros((grid_size , grid_size, nb_anchors, 5+nb_classes))
    cell_w, cell_h = input_size/grid_size, input_size/grid_size
    for i, box in enumerate(boxes):
        x,y,w,h = box
        center_x, center_y = x+w/2, y+h/2
        center_x, center_y = center_x/cell_w, center_y/cell_h
        grid_x = int(np.floor(center_x))
        grid_y = int(np.floor(center_y))
        
        if grid_x < grid_size and grid_y < grid_size:
            boxes_tensor[grid_y, grid_x, :, 0:4] = torch.tensor(nb_anchors * [[center_x-grid_x,center_y-grid_y,w,h]])
            boxes_tensor[grid_y, grid_x, :, 4]  = torch.tensor(nb_anchors * [1.])
    return boxes_tensor



def output_tensor_to_boxes(boxes_tensor, anchors, output_thresh = 0.6, grid_size=grid_size, nb_anchors=nb_anchors, input_size=input_size, nb_classes=nb_classes):
    '''
    Convert output tensor to bounding boxes.
    
    Args:
        boxes_tensor (torch.Tensor): Output tensor containing bounding box information.
        anchors (list): List of anchor box sizes.
        output_thresh (float): Threshold for considering object presence.
        grid_size (int): Size of the grid used for dividing the image.
        nb_anchors (int): Number of anchors.
        input_size (int): Size of the input image.
        nb_classes (int): Number of classes.
        
    Returns:
        numpy.ndarray: Bounding boxes extracted from the output tensor.
    '''
    cell_w, cell_h = input_size/grid_size, input_size/grid_size
    boxes = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            for b in range(nb_anchors):
                anchor_wh = torch.tensor(anchors[b])
                data = boxes_tensor[i,j,b]
                xy = torch.sigmoid(data[:2])
                wh = ((torch.sigmoid(data[2:4]) * 2) ** 1.6) * anchor_wh

                obj_prob = torch.sigmoid(data[4:5])
                
                if obj_prob > output_thresh:
                    x_center, y_center, w, h = xy[0], xy[1], wh[0], wh[1]
                    x, y = (x_center+j)*cell_w-w/2, (y_center+i)*cell_h-h/2
                    box = [x,y,w,h, obj_prob]
                    boxes.append(box)   

    return torch.Tensor(boxes).numpy()

def compute_output_iou(pred_boxes, true_boxes, iou_threshold):

    '''
    Compute the Intersection over Union (IoU) between predicted boxes and ground truth boxes to compute which boxes are well predicted.

    Args:
        pred_boxes (np.ndarray): Predicted bounding boxes.
        true_boxes (np.ndarray): Ground truth bounding boxes.
        iou_threshold (float): IoU threshold to consider a prediction as true positive.

    Returns:
        tuple:
            - pred_y (list): Binary labels for predicted boxes (1 for true positive, 0 otherwise).
            - true_y (list): Binary labels for ground truth boxes (1 for present, 0 otherwise).
            - wh_boxes_well_predicted (list): Width and height of well predicted bounding boxes.
            - wh_boxes (list): Width and height of all ground truth bounding boxes.
    '''
    pred_y = []
    true_y = []
    wh_boxes_well_predicted = []
    wh_boxes = []
    nb_cars = len(true_boxes)
    if len(pred_boxes) > 0:
        sorted_indices = np.argsort(pred_boxes[:, 4])[::-1]
        pred_boxes = pred_boxes[sorted_indices]

    for i, pred_box in enumerate(pred_boxes):
        iou_list = []
        for j, true_box in enumerate(true_boxes):
            iou = calculate_iou(pred_box, true_box)
            iou_list.append(iou)

        if len(iou_list) > 0 and max(iou_list) > iou_threshold:
            true_y.append(1)
            pred_y.append(1)
            wh_boxes_well_predicted.append(list(true_box[2:4]))
        else:
            true_y.append(0)
            pred_y.append(1)
    
    if sum(true_y) < nb_cars:
        cars_not_predicted = nb_cars - sum(true_y)
        true_y += [1 for i in range(cars_not_predicted)]
        pred_y += [0 for i in range(cars_not_predicted)]

    for j, true_box in enumerate(true_boxes):
        wh_boxes.append(list(true_box[2:4]))

    return pred_y, true_y, wh_boxes_well_predicted, wh_boxes


def calculate_iou(box_1, box_2):
    '''
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box_1 (list): Bounding box coordinates [x, y, width, height].
        box_2 (list): Bounding box coordinates [x, y, width, height].

    Returns:
        float: Intersection over Union (IoU) score.
    '''

    box_1 = [[box_1[0], box_1[1]],[box_1[0], box_1[1]+box_1[3]], [box_1[0] + box_1[2], box_1[1]+box_1[3]],[box_1[0] + box_1[2], box_1[1]]]
    box_2 = [[box_2[0], box_2[1]],[box_2[0], box_2[1]+box_2[3]], [box_2[0] + box_2[2], box_2[1]+box_2[3]],[box_2[0] + box_2[2], box_2[1]]]
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    if poly_1.within(poly_2) or poly_2.within(poly_1):
        return 1
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

def nonmax_suppression(boxes, IOU_THRESH = 0.3):
    '''
    Perform non-maximum suppression (NMS) on bounding boxes.

    Args:
        boxes (list): List of bounding boxes.
        IOU_THRESH (float): IoU threshold for suppression (default is 0.3).

    Returns:
        np.ndarray: Bounding boxes after non-maximum suppression.
    '''
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    for i, current_box in enumerate(boxes):
        if current_box[4] <= 0:
            continue
        for j in range(i+1, len(boxes)):
            iou = calculate_iou(current_box, boxes[j])
            if iou > IOU_THRESH:
                boxes[j][4] = 0
    boxes = [box for box in boxes if box[4] > 0]
    return np.array(boxes)

def ap_computation(labels, predictions):
    '''
    Compute the average precision (AP) given ground truth labels and predictions.

    Args:
        labels (list): Ground truth labels (1 for positive, 0 for negative).
        predictions (list): Predicted scores.

    Returns:
        float: Average precision (AP).
    '''
    precision = []
    recall = []
    TP = 0
    FP = 0
    total_positive = np.sum(np.array(labels) == 1)
    for i in range(len(predictions)):
        if predictions[i] == 1 and labels[i] == 1:
            TP += 1
        elif predictions[i] == 1 and labels[i] == 0:
            FP += 1
        if TP+FP > 0:
            precision.append(TP/(TP+FP))
        else:
            precision.append(0)
        recall.append(TP/total_positive)


    recall_values = np.array(np.linspace(0,1,11))
    interpolated_precision = np.array([np.max(np.array(precision)[np.where((np.array(recall) >= thresh))[0]]) if len(np.array(precision)[np.where((np.array(recall) >= thresh))[0]]) > 0 else 0 for thresh in recall_values])

    ap = 1/11 * interpolated_precision.sum()
    return ap


def find_anchors():
    '''
    Computation of the anchor boxes
    '''
    transforms = A.Compose([
        A.Resize(height=416, width=416),
        A.RandomSizedCrop(min_max_height=(int(0.95*416), 416), height=416, width=416, p=0.4),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(p=1.0)
        ],
        bbox_params={
            "format":"coco",
            'label_fields': ['labels']
            })

    with open('data_yolo/dictionnary_bounding_boxes.pkl', 'rb') as fp:
        data_bounding_boxes = pickle.load(fp)

    bbox_wh = []

    for key in list(data_bounding_boxes.keys()):
        image = Image.open("data_yolo/training_images/" + key).convert("RGB")
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
        boxes = data_bounding_boxes[key]["boxes"]
        box_nb = data_bounding_boxes[key]["nb_boxes"]
        labels = torch.zeros((box_nb, 2), dtype=torch.int64)
        labels[:, 0] = 1  
        sample = transforms(**{
                    "image":image,
                    "bboxes": boxes,
                    "labels": labels,
                })
        if len(sample['bboxes']) > 0:
            for bbox in sample['bboxes']:
                x,y,w,h = bbox
                bbox_wh.append((w,h))

    bbox_wh = np.array(bbox_wh)

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(bbox_wh)

    # Clusters centroids
    anchors = kmeans.cluster_centers_

    print("Dimensions des boîtes d'ancrage optimales :", anchors)

    plt.scatter(bbox_wh[:, 0], bbox_wh[:, 1], alpha=0.5, edgecolors='w')
    plt.scatter(anchors[:, 0], anchors[:, 1], marker='X', s=200, linewidths=2, color='red')
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    return anchors.tolist()


def plot_width_height_graph(bboxes_wh, bboxes_wh_well_predicted):
    '''
    Plot a graph of the bounding boxes of cars given their sizes (height, width).
    In green its the cars well predicted (bboxes_wh_well_predicted), in red the cars not well predicted (bboxes_wh - bboxes_wh_well_predicted)
    '''
    bboxes_wh = np.array(bboxes_wh)
    bboxes_wh_well_predicted = np.array(bboxes_wh_well_predicted)

    x = np.linspace(0, 200, 400)
    y = np.linspace(0, 160, 400)

    X, Y = np.meshgrid(x, y)

    Z = X * Y

    mask_small_obj = Z < 32**2
    mask_medium_obj =  (32**2 <= Z) & (Z < 96**2)
    mask_large_obj = 96**2 <= Z 

    plt.figure(figsize=(8, 8))

    plt.pcolormesh(X, Y, mask_small_obj, cmap='viridis', shading='auto', alpha=0.5)
    plt.pcolormesh(X, Y, mask_medium_obj, cmap='viridis', shading='auto', alpha=0.5)
    plt.pcolormesh(X, Y, mask_large_obj, cmap='viridis', shading='auto', alpha=0.5)


    plt.scatter(bboxes_wh[:, 0], bboxes_wh[:, 1], alpha=0.5, edgecolors='w', color='red')
    plt.scatter(bboxes_wh_well_predicted[:, 0], bboxes_wh_well_predicted[:, 1], linewidths=1,edgecolors='w', color='green')
    plt.title('Localisation of well predicted boxes')
    plt.xlabel('Width')
    plt.ylabel('Height')

    plt.show()


def mean_avg_precision(model,device,test_loader, iou_threshold, anchors):
    '''
    Calculate the mean Average Precision (mAP) for object detection models.

    Args:
        model (torch.nn.Module): The object detection model to evaluate.
        device (torch.device): The device to perform computations on.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        iou_threshold (float): The IoU threshold for considering a detection as correct.
        anchors (list): List of anchor box sizes.

    Returns:
        tuple: A tuple containing:
            - AP (float): Mean Average Precision (mAP) score.
            - precision (float): Precision score.
            - recall (float): Recall score.
            - wh_boxes_well_predicted (list): Width and height of well predicted bounding boxes.
            - wh_boxes (list): Width and height of all ground truth bounding boxes.
    '''
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

            pred_boxes = output_tensor_to_boxes(output_tensor, anchors)
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
