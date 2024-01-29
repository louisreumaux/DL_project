import torch
import os
from torch.utils.data import DataLoader
from darknet_training import Darknet19, CustomDataset
import torch.nn.functional as F
import numpy as np

"""
This script loads a pre-trained Darknet19 model, evaluates it on a test dataset,
and computes the accuracy of the model's predictions.
"""

model = Darknet19()

# Load the weights from the checkpoint file
checkpoint_path = 'darknet19_classification_model_finetuned.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)

model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)


test_image_paths = np.load('data_darknet/test/test_image_paths.npy', allow_pickle=True)
test_labels = np.load('data_darknet/test/test_labels.npy', allow_pickle=True)


test_dataset = CustomDataset(test_image_paths, test_labels)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)




def accuracy_computation():
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = F.sigmoid(outputs)
            predicted_classes = torch.round(probabilities)
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted_classes[i].item() == label)
                class_total[label] += 1

    classes = ('other', 'car')

    for i in range(2):
        print('Accuracy of %5s : %.2f %%' %
            (classes[i], 100 * class_correct[i] / class_total[i]))


    correct = class_correct[0] + class_correct[1]
    total = class_total[0] + class_total[1]
    print('Accuracy of the network on the test images: %.2f %%' %
        (100 * correct / total))

accuracy_computation()