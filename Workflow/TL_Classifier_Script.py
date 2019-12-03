# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Import relevant packages
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import torch.optim as optim
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import os
import copy
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)


# %%
# Flags
DISABLE_CUDA = False


# %%
# Hyperparameters
input_dim = 224
train_test_ratio = 0.8

# Declare important file paths
notebook_path = os.path.abspath("TL_Classifier.ipynb")
data_path = os.path.dirname(notebook_path) + '/Colored Dataset/'


# %%
# Select accelerator device
def get_default_device():
    if not DISABLE_CUDA and torch.cuda.is_available():
        print("Running on CUDA!")
        return torch.device('cuda'), True
    else:
        print("Running on CPU!")
        return torch.device('cpu'), False
device, using_cuda = get_default_device()


# %%
# Transform the data
transform = transforms.Compose([
                    transforms.Resize((input_dim, input_dim)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(30),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create training/testing dataloaders
full_set = datasets.ImageFolder(root=data_path, transform=transform)
train_size = int(train_test_ratio * len(full_set))
val_size = int((len(full_set) - train_size) / 2)
test_size = len(full_set) - train_size - val_size
train_set, val_set, test_set = torch.utils.data.random_split(full_set, [train_size, val_size, test_size])

dataset_sizes = {'train': train_size,
                 'val': val_size,
                 'test': test_size}
dataloaders = {'train': torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=1),
               'val': torch.utils.data.DataLoader(val_set, shuffle=False, batch_size=1),
               'test': torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=1)}

class_names = full_set.classes
print (class_names)


# %%
def train_model(model, criterion, optimizer, num_epochs, model_path, confusion_matrix):
    train_accuracy_list = []
    val_accuracy_list = []
    test_accuracy_list = []
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []

    max_test_acc = float("-inf")

    for epoch in range(num_epochs):
        predlist = []
        lbllist = []

        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'): 
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':  # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() 
                running_corrects += (preds == labels.data).sum().item()

                if phase == 'test':
                    predlist.append(preds.numpy()[0])
                    lbllist.append(labels.data.numpy()[0])

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if phase == 'train':
                train_accuracy_list.append(epoch_acc)
                train_loss_list.append(epoch_loss)
            elif phase == 'val':
                val_accuracy_list.append(epoch_acc)
                val_loss_list.append(epoch_loss)
            else:
                if epoch_acc > max_test_acc:
                    max_test_acc = epoch_acc
                    torch.save(model, model_path)

                    rows = zip(predlist,lbllist)
                    with open(confusion_matrix, "w") as f:
                        writer = csv.writer(f)
                        for row in rows:
                            writer.writerow(row)

                test_accuracy_list.append(epoch_acc)
                test_loss_list.append(epoch_loss)               
    return train_accuracy_list, val_accuracy_list, test_accuracy_list, train_loss_list, val_loss_list, test_loss_list, model


# %%
def run_experiment(lr, num_unfreeze, num_epochs, weight_decay, model_filename, confusion_matrix):
    # model_conv = torchvision.models.resnet18(pretrained=True)  # download ResNet18
    model_conv = torchvision.models.googlenet(pretrained=True)  # download GoogLeNet
    for i, param in enumerate(model_conv.parameters()):
        if i < 60 - num_unfreeze:  
            param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 64) 
    model_conv.fc2 = nn.Linear(64, 32) 
    model_conv.fc3 = nn.Linear(32, 5)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.Adam(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=lr, weight_decay=weight_decay)

    return train_model(model_conv, criterion, optimizer_conv, num_epochs, model_filename, confusion_matrix)


# %%

def write_experiment_results_to_file(filename, results_dict):
    with open(filename, 'w+') as file:
        writer = csv.writer(file)
        writer.writerow(results_dict.keys())
        num_rows = len(list(results_dict.values())[0])
        for i in range(num_rows):
            row = []
            for key in results_dict.keys():
                row.append(float(results_dict[key][i]))
            writer.writerow(row)


# %%
lr_list = [0.001, 0.0001]
num_epochs = 100
for lr in lr_list:
    for num_unfreeze in range(3):
        for wd in [0, 1e-5]:
            model_filename = os.path.dirname(notebook_path) + "/experiments/models/GoogLeNet/lr={}_num_unfroze={}_epochs={}_wd={}_COLORED.pth".format(lr, num_unfreeze, num_epochs, wd)
            confusion_matrix = os.path.dirname(notebook_path) + "/experiments/models/GoogLeNet/conf_mat/lr={}_num_unfroze={}_epochs={}_wd={}_COLORED.csv".format(lr, num_unfreeze, num_epochs, wd)
            train_accuracy_list, val_accuracy_list, test_accuracy_list, train_loss_list, val_loss_list, test_loss_list, model = run_experiment(lr=lr, num_unfreeze=num_unfreeze, num_epochs=num_epochs, weight_decay=wd, model_filename=model_filename, confusion_matrix=confusion_matrix)

            results_filename = os.path.dirname(notebook_path) + "/experiments/csv_files/GoogLeNet/lr={}_num_unfroze={}_epochs={}_wd={}_COLORED.csv".format(lr, num_unfreeze, num_epochs, wd)
            results_dict = {"train_accuracy": train_accuracy_list, "val_accuracy": val_accuracy_list, "test_accuracy": test_accuracy_list, "train_loss": train_loss_list, "val_loss": val_loss_list, "test_loss": test_loss_list}
            write_experiment_results_to_file(results_filename, results_dict)

# %%


