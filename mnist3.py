#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:30:40 2023

@author: paularoth
"""
#from einops.layers import torch
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(123)

data_dir = "../data"
batch_size = 64


train_dataset=datasets.MNIST(data_dir, train = True, download = True)

# Transform to Tensor
#train_transform = transforms.Compose([transforms.ToTensor(),])
#train_dataset.transform = train_transform

train_data, val_data, test_data = random_split(train_dataset, [0.8,0.1,0.1])

train_normal_array = []
test_anom_array = []
test_normal_array = []
val_array = []

# Remove all numbers besides 2 from train_data and val_data
# all other numbers are considered as the anomaly numbers
train_data = [data for data in train_data if data[1] == 2]
for i in range(0,len(train_data)):
    train_normal_array.append(train_data[i][0])

val_data = [data for data in val_data if data[1] == 2]
for i in range(0,len(val_data)):
    val_array.append(val_data[i][0])

# We're testing with all numbers
test_anom_data = [data for data in test_data if data[1] != 2]
test_normal_data = [data for data in test_data if data[1] == 2]
for i in range(0,len(test_anom_data)):
    test_anom_array.append(test_anom_data[i][0])
for i in range(0,len(test_normal_data)):
    test_normal_array.append(test_normal_data[i][0])

class MNist:
    def __init__(self, batch_size):
        self.batch = batch_size

        torch.manual_seed(123)

        ## Image Transformation ##
        T = transforms.Compose([
            #transforms.ToPILImage(),   ## Wozu??
            transforms.Resize((30, 30)),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            #            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_normal_image = torch.stack([T(i) for i in train_normal_array])
        test_anom_image = torch.stack([T(i) for i in test_anom_array])
        test_normal_image = torch.stack([T(i) for i in test_normal_array])

        # the mask marks the anomaly patches of the image
        # there are no anomaly patches fpr the normal data
        train_normal_mask = torch.zeros(train_normal_image.size(0), 1, train_normal_image.size(2),train_normal_image.size(3))
        test_normal_mask = torch.zeros(test_normal_image.size(0), 1, test_normal_image.size(2),test_normal_image.size(3))

        # Uses ground-truth images
        # test_anom_mask = torch.stack([Process_mask(T(i)) for i in test_anom_mask_images])
        # First try: implementing analogue to train_normal_mask --> doesn't make sence in the context of the mask
        # Second Try: implementing as ones tensor, because the ground trueth doesn't exist
        test_anom_mask = torch.ones(test_anom_image.size(0), 1, test_anom_image.size(2), test_anom_image.size(3))

        train_normal = tuple(zip(train_normal_image, train_normal_mask))
        test_anom = tuple(zip(test_anom_image, test_anom_mask))
        test_normal = tuple(zip(test_normal_image, test_normal_mask))
        print(f' --Size of train loader: {train_normal_image.size()}--')
        if test_anom_image.size(0) == test_anom_mask.size(0):
            print(f' --Size of test anomaly loader: {test_anom_image.size()}--')
        else:
            print(
                f'[!Info] Size Mismatch between Anomaly images {test_anom_image.size()} and Masks {test_anom_mask.size()} Loaded')
        print(f' --Size of test normal loader: {test_normal_image.size()}--')


        print(f" --Total Image in Validation loader: {len(val_array)}--")

        ####  Final Data Loader ####
        self.train_loader = torch.utils.data.DataLoader(train_normal, batch_size=batch_size, shuffle=True)
        self.test_anom_loader = torch.utils.data.DataLoader(test_anom, batch_size=batch_size, shuffle=False)
        self.test_norm_loader = torch.utils.data.DataLoader(test_normal, batch_size=batch_size, shuffle=False)
        self.validation_loader = torch.utils.data.DataLoader(val_array, batch_size=batch_size, shuffle=False)
        self.test_data = test_data


if __name__ == "__main__":  #wird nur ausgeführt, wenn die Funktion direkt aufgerufen wird

    train = MNist(1)
    for i, j in train.test_anom_loader:
        print(i.shape)
        plt.imshow(i.squeeze(0).permute(1, 2, 0))
        plt.show
        break