import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from utils import boxes_to_tensor



class CarDataset(torch.utils.data.Dataset):
    '''
    Custom dataset class for handling image data with associated labels.
    '''
    def __init__(self, data, input_size, images_folder, data_augmentation = True):
        self.data = data,
        self.images_folder = images_folder
        self.data_aug = data_augmentation
        self.transform_image_w_bboxes = A.Compose([
        A.Resize(height=input_size, width=input_size),
        A.RandomSizedCrop(min_max_height=(int(0.95*input_size), input_size), height=input_size, width=input_size, p=0.4),
        A.HorizontalFlip(p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
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
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
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
                target_tensor = boxes_to_tensor(boxes.type(torch.float32), labels)
            else:
                target_tensor = torch.zeros((13 ,13, 5, 5))

        else:
            target_tensor = torch.zeros((13 ,13, 5, 5))
            image = self.transform_image_wout_bboxes(image=image)["image"] if self.data_aug else self.transform_image_wout_bboxes_wout_geom(image=image)["image"]

        return image.float() / 255.0, target_tensor


