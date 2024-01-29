from PIL import Image
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
    '''
    Custom dataset class for handling image data with associated labels.
    '''
    def __init__(self, image_paths, labels, input_size=224):
        self.image_paths = image_paths
        self.labels = labels

        self.transforms = A.Compose([
        A.Resize(height=input_size, width=input_size),
        A.RandomSizedCrop(min_max_height=(int(0.95*input_size), input_size), height=input_size, width=input_size, p=0.4),
        A.HorizontalFlip(p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ToTensorV2(p=1.0)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        label = self.labels[idx]

        image = self.transforms(image=image)["image"]

        image = image.to(torch.float32)  
        
        return image, label