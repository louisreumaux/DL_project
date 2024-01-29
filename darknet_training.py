import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from net.darknet_net import Darknet19
from darknet_dataset import CustomDataset
import json
import numpy as np

'''
This script trains a Darknet19 model on a custom dataset and saves the trained model weights.
'''

with open('parameters/darknet_parameters.json', 'r') as f:
    parameters = json.load(f)

# Hyperparameters
learning_rate = parameters["learning_rate"]
learning_rate_finetuning = parameters["learning_rate_finetuning"]
weight_decay = parameters["weight_decay"]
momentum = parameters["momentum"]
num_epochs = parameters["epochs"]
num_epochs_finetuning = parameters["epochs_finetuning"]
batch_size = parameters["batch_size"]



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Darknet19()
print(model)

### TRAIN ###

train_image_paths = np.load('data_darknet/train/train_image_paths.npy', allow_pickle=True)
train_labels = np.load('data_darknet/train/train_labels.npy', allow_pickle=True)

train_dataset = CustomDataset(train_image_paths, train_labels,input_size=224)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_dataset_finetuning = CustomDataset(train_image_paths, train_labels, input_size=448)
train_loader_finetuning = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(
), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)


# Training loop
def train_model(model, num_epochs, saving_path, train_loader):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        start_time = time.time()

        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            labels = labels.long()

            loss = criterion(outputs, labels.float())

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        end_time = time.time()
        epoch_time = end_time - start_time

        average_loss = running_loss / len(train_loader)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, Time: {epoch_time:.2f} seconds')

    torch.save(model.state_dict(), saving_path)

if __name__ == "__main__":
    weight_folder = "net/darknet_weights/"
    saving_path = weight_folder + "darknet19_classification_model.pth"
    saving_path_finetuned = weight_folder + "darknet19_classification_model_finetuned.pth"
    train_model(model, num_epochs, saving_path, train_loader)
    for g in optimizer.param_groups:
        g['lr'] = learning_rate_finetuning
    train_model(model, num_epochs_finetuning, saving_path_finetuned, train_loader_finetuning)
