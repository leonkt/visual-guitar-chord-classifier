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
import os
import copy
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

# Flags
DISABLE_CUDA = False

# Hyperparameters
input_dim = 224
train_test_ratio = 0.8

# Declare important file paths
notebook_path = os.path.abspath("TL_Classifier.ipynb")
data_path = os.path.dirname(notebook_path) + '/Colored Dataset/'

# Select accelerator device
def get_default_device():
    if not DISABLE_CUDA and torch.cuda.is_available():
        print("Running on CUDA!")
        return torch.device('cuda'), True
    else:
        print("Running on CPU!")
        return torch.device('cpu'), False
device, using_cuda = get_default_device()

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
               'val': torch.utils.data.DataLoader(val_set, shuffle=True, batch_size=1),
               'test': torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=1)}

class_names = full_set.classes
print (class_names)

########################################################################################################

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

########################################################################################################

model = torch.load(os.path.dirname(notebook_path) + '/experiments/models/lr=0.001_num_unfroze=0_epochs=1_wd=0_COLORED_NOADD.pth', map_location=device)
model.eval()
# visualize_model(model)

# from PIL import Image
# import torchvision.transforms.functional as F
# from flashtorch.utils import load_image
# from flashtorch.saliency import Backprop
# from flashtorch.activmax import GradientAscent

# def apply_transforms(image, size=224):
#     if not isinstance(image, Image.Image):
#         image = F.to_pil_image(image)

#     tensor = transform(image).unsqueeze(0)
#     tensor.requires_grad = True
#     return tensor

# for i, (example, label) in enumerate(dataloaders['train']):  
#     image = example[0]
#     label = label[0]
#     break

# backprop = Backprop(model)
# model.eval()
# transformed = apply_transforms(image)
# backprop.visualize(transformed, 0, guided=True)
# plt.show()



########################################################################################################
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np 

predlist = []
lbllist = []

with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['test']):
        inputs = inputs.to(device)
        classes = classes.float().numpy()[0]
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        # Append batch prediction results
        predlist.append(preds)
        lbllist.append(classes)

# Confusion matrix
print (lbllist)
print (predlist)

labels = ['C', 'D', 'Em', 'F', 'G']

conf_mat=confusion_matrix(np.array(lbllist), np.array(predlist))
df_cm = pd.DataFrame(conf_mat, range(5), range(5))

ax= plt.subplot()
sn.heatmap(conf_mat, annot=True, ax=ax)

ax.set_xlabel('Predicted Labels');ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)


plt.show()
print (conf_mat)
########################################################################################################