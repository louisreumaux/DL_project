import time
import torch
import json
from torch.utils.data import DataLoader

import pickle
from yolo_dataset import CarDataset
from yolo_loss import YoloLoss, YoloLoss2
from net.darknet_net import Darknet19
from net.yolo_net import YOLO
import math
import matplotlib.pyplot as plt
from utils import find_anchors

with open('parameters/yolo_parameters.json', 'r') as f:
    parameters = json.load(f)

if len(parameters["anchors"]) > 0:
    anchors = parameters["anchors"]
else:
    anchors = find_anchors()
    parameters["anchors"] = anchors
    with open('parameters/yolo_parameters.json', 'w') as f:
        json.dump(parameters, f)

num_classes = parameters["classes"]
learning_rate = parameters["learning_rate"]
weight_decay = parameters["weight_decay"]
momentum = parameters["momentum"]
num_epochs = parameters["epochs"]

lambda_xy = parameters["lambda_xy"]
lambda_wh = parameters["lambda_wh"]
lambda_conf = parameters["lambda_conf"]
lambda_noobj = parameters["lambda_noobj"]
batch_size = parameters["batch_size"]
images_folder = "data_yolo/training_images/"

# Instantiate the model
model = YOLO(num_classes,anchors)
darknet19 = Darknet19()

# Charger les poids de Darknet19 pré-entraîné

yolo_weight_folder = "net/yolo_weights/"
darknet_weight_folder = "net/darknet_weights/"
darknet19_weights_path = "darknet19_classification_model_finetuned.pth"
darknet19.load_state_dict(torch.load(darknet_weight_folder + darknet19_weights_path))

# Transférer les poids jusqu'à une couche commune (à ajuster selon votre architecture)
model.features_1.load_state_dict(darknet19.features_1.state_dict())
model.features_2.load_state_dict(darknet19.features_2.state_dict())
model.max_pool.load_state_dict(darknet19.max_pool.state_dict())

# Print the model summary
print(model)

with open('data_yolo/dictionnary_bounding_boxes.pkl', 'rb') as fp:
    data_bounding_boxes = pickle.load(fp)

# Créez une instance de votre dataset

train_dataset = CarDataset(data=data_bounding_boxes, input_size=416, images_folder=images_folder)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Définition de la fonction de perte et de l'optimiseur
criterion = YoloLoss(num_classes, device, lambda_xy=lambda_xy, lambda_wh=lambda_wh, lambda_conf=lambda_conf, lambda_noobj=lambda_noobj, anchors=anchors)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, nesterov=True)

model = model.to(device)

def train_model(num_epochs, saving_path = "yolo_model.pth"):
    losses = []
    epochs = []
    log_losses = []
    best_loss = 1e7

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        xy_loss = 0.0
        wh_loss = 0.0
        obj_loss = 0.0
        noobj_loss = 0.0

        start_time = time.time()

        iters = 0

        if (epoch == 320) or (epoch == 460):
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*0.2
            
        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            if (epoch == 0) & (iters <= len(train_loader)):
                power = 4
                lr = learning_rate * (iters / (len(train_loader))) ** power
                for g in optimizer.param_groups:
                    g['lr'] = lr
            

            # Mise à zéro des gradients
            optimizer.zero_grad()

            # Passage avant
            outputs = model(inputs)


            # Calcul de la perte
            loss,loss_detail = criterion(outputs, labels)

            # Rétropropagation des gradients
            loss.backward()

            # Mise à jour des poids
            optimizer.step()
 
            running_loss += loss.item()
            xy_loss += loss_detail[0]
            wh_loss += loss_detail[1]
            obj_loss += loss_detail[2]
            noobj_loss +=loss_detail[3]

            iters += 1

        #lr_scheduler.step()
        end_time = time.time()
        epoch_time = end_time - start_time

        # Affichage de la perte moyenne par époque
        average_loss = running_loss / len(train_loader)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, Time: {epoch_time:.2f} seconds')
        print(f'Epoch {epoch + 1}/{num_epochs}, Weighted XY loss: {xy_loss:.4f}, Weighted WH loss: {wh_loss:.4f}, Weighted Object loss: {obj_loss:.4f}, Weighted No object loss: {noobj_loss:.4f}')
        print("\n")
        
        epochs.append(epoch + 1)
        log_losses.append(math.log2(running_loss))
        losses.append(running_loss)

        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(model.state_dict(), saving_path)

    plt.plot(epochs, losses, marker='o')
    plt.title('Loss evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(epochs, log_losses, marker='o')
    plt.title('Loss evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


if __name__ == "__main__":
    saving_path = yolo_weight_folder + "yolo_model_v2.pth"
    train_model(num_epochs, saving_path)


