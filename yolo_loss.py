import torch
import torch.nn as nn
import math

class YoloLoss(nn.Module):

    def __init__(self, num_classes, device, grid_size = 13, anchors = 5, lambda_xy = 10, lambda_wh = 0.1, lambda_conf = 10, lambda_noobj = 1):
        super(YoloLoss, self).__init__()
        
        self.grid_size = grid_size
        self.anchors = anchors
        self.device = device
        self.num_anchors = len(self.anchors)
        self.num_classes = num_classes
        self.lambda_xy = lambda_xy
        self.lambda_wh = lambda_wh
        self.lambda_conf = lambda_conf
        self.lambda_noobj = lambda_noobj
        

    def forward(self, predictions, target):

        predictions = predictions.permute(0,2,3,1)
        W_grid, H_grid = self.grid_size, self.grid_size
        predictions = predictions.view(-1, H_grid, W_grid, self.num_anchors, 4+1+self.num_classes)

        pred_xy, pred_wh, pred_obj_prob, _ = post_process_output(predictions, self.anchors, self.num_anchors, self.device)
        true_xy, true_wh, true_obj_prob, _ = post_process_target(target)
        
        # Calculate areas of pred_bbox
        pred_ul = pred_xy - 0.5*pred_wh
        pred_br = pred_xy + 0.5*pred_wh
        pred_area = pred_wh[:,:,:,:,0]*pred_wh[:,:,:,:,1]
        
        # Calculate areas of true_bbox
        true_ul = true_xy - 0.5*true_wh
        true_br = true_xy + 0.5*true_wh
        true_area = true_wh[:,:,:,:,0]*true_wh[:,:,:,:,1]

        # Calculate IOU between each pred_bbox and corresponding true_bbox (within the same cell)
        intersect_ul = torch.max(pred_ul, true_ul)
        intersect_br = torch.min(pred_br, true_br)
        intersect_wh = intersect_br - intersect_ul
        intersect_area = intersect_wh[:,:,:,:,0]*intersect_wh[:,:,:,:,1]
        
        # In each cell, determine best_box - the box with the highest IOU with true_bbox among the remaining 4 pred_bbox
        iou = intersect_area/(pred_area + true_area - intersect_area)
        max_iou = torch.max(iou, dim=3, keepdim=True)[0]
        best_box_index =  torch.unsqueeze(torch.eq(iou, max_iou).float(), dim=-1)
        true_box_conf = best_box_index*true_obj_prob
        
        # Calculate individual losses according to the formulas in the image
        xy_loss =  (square_error(pred_xy, true_xy)*true_box_conf).sum()
        wh_loss =  (square_error(pred_wh, true_wh)*true_box_conf).sum()
        obj_loss = (square_error(pred_obj_prob, true_obj_prob)*true_box_conf).sum()
        noobj_loss = (square_error(pred_obj_prob, true_obj_prob)*(1-true_box_conf)).sum()

        total_loss = self.lambda_xy*xy_loss + self.lambda_wh*wh_loss + self.lambda_conf*obj_loss + self.lambda_noobj*noobj_loss

        return total_loss, [self.lambda_xy*xy_loss, self.lambda_wh*wh_loss, self.lambda_conf*obj_loss, self.lambda_noobj*noobj_loss]


def post_process_output(output, anchors, num_anchors, device):
    """Convert output of model to pred_xywh"""
    # xy
    xy = torch.sigmoid(output[:,:,:,:,:2])

    # wh
    wh = output[:,:,:,:,2:4]
    
    anchors_wh = torch.Tensor(anchors).view(1,1,1,num_anchors,2).to(device)
    wh = ((torch.sigmoid(wh) * 2) ** 1.6) * anchors_wh

    # objectness confidence
    obj_prob = torch.sigmoid(output[:,:,:,:,4:5])
    
    return xy, wh, obj_prob, 0

def post_process_target(target_tensor):
    """
    Separate the target tensor into individual components: xy, wh, object probability, class distribution.
    """
    xy = target_tensor[:,:,:,:,:2]
    wh = target_tensor[:,:,:,:,2:4]
    obj_prob = target_tensor[:,:,:,:,4:5]

    return xy, wh, obj_prob, 0

def square_error(output, target):
    return (output-target)**2



class YoloLoss2(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    (1,1,1,10,0.1)
    """

    def __init__(self, num_classes, device, grid_size = 13, anchors = 5, lambda_coord_xy = 200, lambda_coord_wh = 0.2, lambda_coord_noobj_xy = 0.05, lambda_coord_noobj_wh = 0.00001, lambda_conf = 200, lambda_noobj = 0.2):
        super(YoloLoss2, self).__init__()
        
        self.grid_size = grid_size
        self.anchors = anchors
        self.device = device
        self.num_anchors = len(self.anchors)
        self.num_classes = num_classes
        self.lambda_coord_xy = lambda_coord_xy
        self.lambda_coord_wh = lambda_coord_wh
        self.lambda_coord_noobj_xy = lambda_coord_noobj_xy
        self.lambda_coord_noobj_wh = lambda_coord_noobj_wh
        self.lambda_conf = lambda_conf
        self.lambda_noobj = lambda_noobj
        self.anchors_tensor_wh = torch.tensor([[[self.anchors] * 13] * 13] * 16)
        self.anchors_tensor_xy = torch.tensor([[[[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]] * 13] * 13] * 16)

    def forward(self, predictions, target):

        predictions = predictions.permute(0,2,3,1)
        W_grid, H_grid = self.grid_size, self.grid_size
        predictions = predictions.view(-1, H_grid, W_grid, self.num_anchors, 4+1+self.num_classes)

        pred_xy, pred_wh, pred_obj_prob, _ = post_process_output(predictions, self.anchors, self.num_anchors, self.device)
        true_xy, true_wh, true_obj_prob, _ = post_process_target(target)
        
        # Calculate areas of pred_bbox
        pred_ul = pred_xy - 0.5*pred_wh
        pred_br = pred_xy + 0.5*pred_wh
        pred_area = pred_wh[:,:,:,:,0]*pred_wh[:,:,:,:,1]
        
        # Calculate areas of true_bbox
        true_ul = true_xy - 0.5*true_wh
        true_br = true_xy + 0.5*true_wh
        true_area = true_wh[:,:,:,:,0]*true_wh[:,:,:,:,1]

        # Calculate IOU between each pred_bbox and corresponding true_bbox (within the same cell)
        intersect_ul = torch.max(pred_ul, true_ul)
        intersect_br = torch.min(pred_br, true_br)
        intersect_wh = intersect_br - intersect_ul
        intersect_area = intersect_wh[:,:,:,:,0]*intersect_wh[:,:,:,:,1]
        
        # In each cell, determine best_box - the box with the highest IOU with true_bbox among the remaining 4 pred_bbox
        iou = intersect_area/(pred_area + true_area - intersect_area)
        max_iou = torch.max(iou, dim=3, keepdim=True)[0]
        best_box_index =  torch.unsqueeze(torch.eq(iou, max_iou).float(), dim=-1)
        true_box_conf = best_box_index*true_obj_prob
        
        # Calculate individual losses according to the formulas in the image
        xy_loss =  (square_error(pred_xy, true_xy)*true_box_conf).sum()
        wh_loss =  (square_error(pred_wh, true_wh)*true_box_conf).sum()
        obj_loss = (square_error(pred_obj_prob*true_box_conf, iou.unsqueeze(-1)*true_box_conf)).sum()
        noobj_loss = (square_error(pred_obj_prob, torch.zeros(pred_obj_prob.shape).to(self.device))*(1-true_box_conf)).sum()
        noobj_xy_loss = (square_error(pred_xy, (self.anchors_tensor_xy).to(self.device))*(1-true_box_conf)).sum()
        noobj_wh_loss = (square_error(pred_wh, (self.anchors_tensor_wh).to(self.device))*(1-true_box_conf)).sum()


        # Combined Loss
        total_loss = self.lambda_coord_xy*xy_loss + self.lambda_coord_wh*wh_loss + self.lambda_conf*obj_loss + self.lambda_noobj*noobj_loss + self.lambda_coord_noobj_xy*noobj_xy_loss + self.lambda_coord_noobj_wh*noobj_wh_loss

        return total_loss, [self.lambda_coord_xy*xy_loss, self.lambda_coord_wh*wh_loss, self.lambda_conf*obj_loss, self.lambda_noobj*noobj_loss, self.lambda_coord_noobj_xy*noobj_xy_loss, self.lambda_coord_noobj_wh*noobj_wh_loss]
