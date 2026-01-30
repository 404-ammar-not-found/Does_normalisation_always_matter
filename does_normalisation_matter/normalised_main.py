import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

import torchinfo
from torchinfo import summary

import matplotlib
import matplotlib.pyplot as plt

import tqdm as tqdm_version_check
from tqdm import tqdm

import pandas as pd
import numpy as np
from pathlib import Path

import re

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from PIL import Image, ImageFile

from sklearn import preprocessing
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms, models

CSV_PATH = Path("data/train_info.csv")
IMAGE_DIR = Path("data/train")

print(f"Torch version : {torch.__version__}")
print(f"Torchvision version : {torchvision.__version__}")
print(f"Torchinfo version : {torchinfo.__version__}")
print(f"Matplotlib version : {matplotlib.__version__}")
print(f"TQDM version : {tqdm_version_check.__version__}")

from generate_dataset import generate_dataset, generate_train_val_test_datasets
from dataset_filteration import filter_small_classes
from generate_dataloader import generate_dataloader
from engine import train_model
from config_parameters import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY, NUM_WORKERS

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


regular_transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

colour_normalised_transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



dataset = generate_dataset()
print("Before:", (dataset.classes), "classes")

filtered_dataset = filter_small_classes(dataset, min_samples=500)
print("After:", (filtered_dataset.classes), "classes")

class_names = dataset.classes
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

train_ds, val_ds, test_ds = generate_train_val_test_datasets(filtered_dataset,
                                                             train_ratio=0.7,
                                                             val_ratio=0.15,
                                                             test_ratio=0.15,
                                                             random_seed=42)


train_dataloader, validation_dataloader, test_dataloader = generate_dataloader(train_ds, 
                                                                               val_ds, 
                                                                               test_ds,
                                                                               BATCH_SIZE=BATCH_SIZE,
                                                                               num_workers=NUM_WORKERS)



model = models.resnet18(pretrained=True)
output_shape = len(class_names)

model.fc = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(
        in_features=model.fc.in_features,
        out_features=output_shape,
        bias=True
    )
)

model = model.to(device)

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

model, best_state_dict = train_model(model, 
                train_dataloader=train_dataloader,
                validation_dataloader=validation_dataloader,
                device=device,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                patience=PATIENCE,
                verbose_ct=VERBOSE_CT,
                num_epochs=NUM_EPOCHS
                )