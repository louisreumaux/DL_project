import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2



class CarDataset(torch.utils.data.Dataset):
    def __init__(self, data, input_size, images_folder, data_augmentation = True):
        self.data = data,
        self.images_folder = images_folder
        self.data_aug = data_augmentation
        self.transform_image_w_bboxes = A.Compose([
        A.Resize(height=input_size, width=input_size),
        A.RandomSizedCrop(min_max_height=(int(0.95*input_size), input_size), height=input_size, width=input_size, p=0.4),
        A.HorizontalFlip(p=0.5),
        #A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ToTensorV2(p=1.0)
        ],
        bbox_params={
            "format":"coco",
            'label_fields': ['labels']
            })
        
        self.transform_image_wout_bboxes = A.Compose([
        A.Resize(height=input_size, width=input_size),
        A.RandomSizedCrop(min_max_height=(int(0.95*input_size), input_size), height=input_size, width=input_size, p=0.4),
        A.HorizontalFlip(p=0.5),
        #A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ToTensorV2(p=1.0)
        ])

        self.transform_image_w_bboxes_wout_geom = A.Compose([
        A.Resize(height=input_size, width=input_size),
        ToTensorV2(p=1.0)
        ],
        bbox_params={
            "format":"coco",
            'label_fields': ['labels']
            })
        
        self.transform_image_wout_bboxes_wout_geom = A.Compose([
        A.Resize(height=input_size, width=input_size),
        ToTensorV2(p=1.0)
        ])
        
    def __len__(self):
        data = self.data[0]
        return len(data.keys())

    def __getitem__(self, idx):
        """"""
        data = self.data[0]
        key_list = list(data.keys())
        image_data = data[key_list[idx]]
        image_path = self.images_folder + image_data["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
        boxes = image_data["boxes"]
        box_nb = image_data["nb_boxes"]

        if box_nb > 0:
            labels = torch.zeros((box_nb, 2), dtype=torch.int64)
            labels[:, 0] = 1        
            sample = self.transform_image_w_bboxes(**{
                "image":image,
                "bboxes": boxes,
                "labels": labels,
            }) if self.data_aug else self.transform_image_w_bboxes_wout_geom(**{
                "image":image,
                "bboxes": boxes,
                "labels": labels,
            })
            image = sample['image']
            if len(sample["bboxes"]) > 0:
                boxes = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                target_tensor = self.boxes_to_tensor(boxes.type(torch.float32), labels)
            else:
                target_tensor = torch.zeros((13 ,13, 5, 5))

        else:
            target_tensor = torch.zeros((13 ,13, 5, 5))
            image = self.transform_image_wout_bboxes(image=image)["image"] if self.data_aug else self.transform_image_wout_bboxes_wout_geom(image=image)["image"]

        return image, target_tensor
    

    def boxes_to_tensor(self, boxes, labels):
        """
        Convert list of boxes (and labels) to tensor format
        Output:
            boxes_tensor: shape = (Batchsize, S, S, Box_nb, (4+1+CLS))
        """

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
                boxes_tensor[grid_y, grid_x, :, 4]  = torch.tensor(BOX * [1.])

        return boxes_tensor



class TestCarDataset(torch.utils.data.Dataset):
    def __init__(self, data, input_size):
        self.data = data,
        self.transform_image_w_bboxes = A.Compose([
        A.Resize(height=input_size, width=input_size),
        ToTensorV2(p=1.0)
        ],
        bbox_params={
            "format":"coco",
            'label_fields': ['labels']
            })
        
        self.transform_image_wout_bboxes = A.Compose([
        A.Resize(height=input_size, width=input_size),
        ToTensorV2(p=1.0)
        ])
        
    def __len__(self):
        data = self.data[0]
        return len(data.keys())

    def __getitem__(self, idx):
        """"""
        data = self.data[0]
        key_list = list(data.keys())
        image_data = data[key_list[idx]]
        image_path = "data_yolo/testing_images/" + image_data["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
        boxes = image_data["boxes"]
        box_nb = image_data["nb_boxes"]

        if box_nb > 0:
            labels = torch.zeros((box_nb, 2), dtype=torch.int64)
            labels[:, 0] = 1        
            sample = self.transform_image_w_bboxes(**{
                "image":image,
                "bboxes": boxes,
                "labels": labels,
            })
            image = sample['image']
            if len(sample["bboxes"]) > 0:
                boxes = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                target_tensor = self.boxes_to_tensor(boxes.type(torch.float32), labels)
            else:
                target_tensor = torch.zeros((13 ,13, 5, 5))

        else:
            target_tensor = torch.zeros((13 ,13, 5, 5))
            image = self.transform_image_wout_bboxes(image=image)["image"]

        return image, target_tensor
    

    def boxes_to_tensor(self, boxes, labels):
        """
        Convert list of boxes (and labels) to tensor format
        Output:
            boxes_tensor: shape = (Batchsize, S, S, Box_nb, (4+1+CLS))
        """

        S = 13
        BOX = 5
        CLS = 0
        W = 416
        H = 416
        boxes_tensor = torch.zeros((S , S, BOX, 5+CLS))
        cell_w, cell_h = W/S, H/S
        for i, box in enumerate(boxes):
            x,y,w,h = box
            # normalize xywh with cell_size
            #x,y,w,h = x/cell_w, y/cell_h, w/cell_w, h/cell_h
            center_x, center_y = x+w/2, y+h/2
            center_x, center_y = center_x/cell_w, center_y/cell_h
            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))
            
            if grid_x < S and grid_y < S:
                boxes_tensor[grid_y, grid_x, :, 0:4] = torch.tensor(BOX * [[center_x-grid_x,center_y-grid_y,w,h]])
                boxes_tensor[grid_y, grid_x, :, 4]  = torch.tensor(BOX * [1.])
                #boxes_tensor[grid_y, grid_x, :, 5:]  = torch.tensor(np.array(BOX*[labels[i].numpy()]))
        return boxes_tensor

