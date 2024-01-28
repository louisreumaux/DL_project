import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from shapely.geometry import Polygon
import pickle
import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from PIL import Image
from sklearn.cluster import KMeans


BOX_COLOR = (0, 0, 255)
S = 13 
BOX = 5 
CLS = 2 
H, W = 416, 416
OUTPUT_THRESH = 0.6
ANCHOR_BOXS = [[
            56.448586925137235,
            42.758431397356944
        ],
        [
            104.07292781091715,
            74.05125952405875
        ],
        [
            73.9442156753153,
            54.508589646511204
        ],
        [
            32.10578740924305,
            37.558116701887016
        ],
        [
            169.09191681233446,
            119.81162735479455
        ]]



def plot_img(img, size=(7,7)):
    plt.figure(figsize=size)
    plt.imshow(img[:,:,::-1])
    plt.show()
    
def visualize_bbox(img, boxes, thickness=2, color=BOX_COLOR, draw_center=True):
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





def target_tensor_to_boxes(boxes_tensor):
    '''
    Recover target tensor (tensor output of dataset) to bboxes
    '''
    cell_w, cell_h = W/S, H/S
    boxes = []
    for i in range(S):
        for j in range(S):
            data = boxes_tensor[i,j,0]
            #x_center,y_center, w, h, obj_prob, cls_prob = data[0], data[1], data[2], data[3], data[4], data[5:]
            #prob = obj_prob*max(cls_prob)
            x_center,y_center, w, h, obj_prob = data[0], data[1], data[2], data[3], data[4]
            prob = obj_prob
            if prob > OUTPUT_THRESH:
                x, y = (x_center+j)*cell_w-w/2, (y_center+i)*cell_h-h/2
                box = [x,y,w,h]
                boxes.append(box)
            
    return torch.Tensor(boxes).numpy()



def boxes_to_tensor(boxes, obj_prob):
    '''
    Build the target tensor from bboxes coordinates
    '''
    S = 13
    BOX = 5
    CLS = 0
    W = 416
    H = 416
    boxes_tensor = torch.zeros((S , S, BOX, 5+CLS))
    cell_w, cell_h = W/S, H/S
    for i, box in enumerate(boxes):
        x,y,w,h = box
        center_x, center_y = x+w/2, y+h/2
        center_x, center_y = center_x/cell_w, center_y/cell_h
        grid_x = int(np.floor(center_x))
        grid_y = int(np.floor(center_y))
        
        if grid_x < S and grid_y < S:
            boxes_tensor[grid_y, grid_x, :, 0:4] = torch.tensor(BOX * [[center_x-grid_x,center_y-grid_y,w,h]])
            boxes_tensor[grid_y, grid_x, :, 4]  = torch.tensor(BOX * [obj_prob])
            #boxes_tensor[grid_y, grid_x, :, 5:]  = torch.tensor(np.array(BOX*[labels[i].numpy()]))
    return boxes_tensor



def output_tensor_to_boxes(boxes_tensor):
    '''
    Recover output tensor to bboxes
    '''
    cell_w, cell_h = W/S, H/S
    boxes = []
    
    for i in range(S):
        for j in range(S):
            for b in range(BOX):
                anchor_wh = torch.tensor(ANCHOR_BOXS[b])
                data = boxes_tensor[i,j,b]
                xy = torch.sigmoid(data[:2])
                wh = ((torch.sigmoid(data[2:4]) * 2) ** 1.6) * anchor_wh

                obj_prob = torch.sigmoid(data[4:5])
                
                if obj_prob > OUTPUT_THRESH:
                    x_center, y_center, w, h = xy[0], xy[1], wh[0], wh[1]
                    x, y = (x_center+j)*cell_w-w/2, (y_center+i)*cell_h-h/2
                    box = [x,y,w,h, obj_prob]
                    boxes.append(box)

    return torch.Tensor(boxes).numpy()

def compute_output_iou(pred_boxes, true_boxes, iou_threshold):
    '''
    In this function, we compute the boxes well predicted as well as the coordinate of the boxes we predicted well
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
    Computation of IoU
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
    Computation of the average precision given the labels and the predictions
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
    Computation of the anchor boxes given the 
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

    # K-means
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


