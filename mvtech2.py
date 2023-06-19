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
import cv2
import os

random.seed(123)

batch_size = 64

product = 'bottle'

root = '/home/roth_pu/RESIKOAST/hotspot-detection/VT-ADL/mvtech_dataset/'

## Train dataset
def Train_data (root = root, product = product):
    train_data = []
    dir = sorted(os.listdir(root))
    for d in dir:
        if product == "all":
            path = root + d + '/train/good/'
            img = sorted(os.listdir(path))
            for i in img:
                train_data.append(cv2.imread(path + i))

        elif product == d:
            path = root + d + '/train/good/'
            img = sorted(os.listdir(path))
            for i in img:
                train_data.append(cv2.imread(path + i))

    return train_data

## Test (norm) dataset
def Test_norm_data (root = root, product = product):
    test_norm_data = []
    dir = sorted(os.listdir(root))
    for d in dir:
        if product == "all":
            path = root + d + '/test/good/'
            img = sorted(os.listdir(path))
            for i in img:
                test_norm_data.append(cv2.imread(path + i))

        elif product == d:
            path = root + d + '/test/good/'
            img = sorted(os.listdir(path))
            for i in img:
                test_norm_data.append(cv2.imread(path + i))

    return test_norm_data

## Test (anom) dataset
def Test_anom_data (root = root, product = product):
    test_anom_data = []
    dir = sorted(os.listdir(root))
    for d in dir:
        damage = sorted(os.listdir(root + d + '/test/'))
        for da in damage:
            if da != "good":
                path = root + d + '/test/' + da + '/'

                if product == "all":
                    img = sorted(os.listdir(path))
                    for i in img:
                        test_anom_data.append(cv2.imread(path + i))

                elif product == d:
                    img = sorted(os.listdir(path))
                    for i in img:
                        test_anom_data.append(cv2.imread(path + i))

    return test_anom_data

## Ground-truth dataset
def Ground_truth_data (root = root, product = product):
    ground_truth_data = []
    dir = sorted(os.listdir(root))
    for d in dir:
        damage = sorted(os.listdir(root + d + '/ground_truth/'))
        for da in damage:
            path = root + d + '/ground_truth/' + da + '/'

            if product == "all":
                img = sorted(os.listdir(path))
                for i in img:
                    ground_truth_data.append(cv2.imread(path + i))

            elif product == d:
                img = sorted(os.listdir(path))
                for i in img:
                    ground_truth_data.append(cv2.imread(path + i))

    return ground_truth_data


def Process_mask(mask):
    mask = np.where(mask > 0., 1, mask)
    return torch.tensor(mask)


def ran_generator(length, shots=1):
    rand_list = random.sample(range(0, length), shots)
    return rand_list

# Transform to Tensor
#train_transform = transforms.Compose([transforms.ToTensor(),])
#train_dataset.transform = train_transform

#train_data, val_data, test_data = random_split(train_dataset, [0.8,0.1,0.1])


class Mvtech:
    def __init__(self, batch_size = 1, root='/home/roth_pu/RESIKOAST/hotspot-detection/VT-ADL/mvtech_dataset/', product=product):
        self.batch = batch_size
        self.root = root
        self.product = product
        torch.manual_seed(123)

        # Importing all the image_path dictionaries for  test and train data #
        train_data = Train_data(root=self.root, product=self.product)
        test_norm_data = Test_norm_data(root=self.root, product=self.product)
        test_anom_data = Test_anom_data(root=self.root, product=self.product)
        ground_truth_data = Ground_truth_data(root=self.root, product=self.product)

        ## Image Transformation ##
        T = transforms.Compose([
            transforms.ToPILImage(),   ## Wozu??
            transforms.Resize((550, 550)),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            #            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_normal_image = torch.stack([T(i) for i in train_data])
        test_normal_image = torch.stack([T(i) for i in test_norm_data])
        test_anom_image = torch.stack([T(i) for i in test_anom_data])

        # the mask marks the anomaly patches of the image
        # there are no anomaly patches fpr the normal data
        train_normal_mask = torch.zeros(train_normal_image.size(0), 1, train_normal_image.size(2),train_normal_image.size(3))
        test_normal_mask = torch.zeros(test_normal_image.size(0), 1, test_normal_image.size(2),test_normal_image.size(3))

        # Uses ground-truth images
        test_anom_mask = torch.stack([Process_mask(T(i)) for i in ground_truth_data])


        train_normal = tuple(zip(train_normal_image, train_normal_mask))
        test_anom = tuple(zip(test_anom_image, test_anom_mask))
        test_normal = tuple(zip(test_normal_image, test_normal_mask))
        print(f' --Size of {self.product} train loader: {train_normal_image.size()}--')
        if test_anom_image.size(0) == test_anom_mask.size(0):
            print(f' --Size of {self.product} test anomaly loader: {test_anom_image.size()}--')
        else:
            print(
                f'[!Info] Size Mismatch between Anomaly images {test_anom_image.size()} and Masks {test_anom_mask.size()} Loaded')
        print(f' --Size of {self.product} test normal loader: {test_normal_image.size()}--')

        # validation set #
        num = ran_generator(len(test_anom), 10)
        val_anom = [test_anom[i] for i in num]
        num = ran_generator(len(test_normal), 10)
        val_norm = [test_normal[j] for j in num]
        val_set = [*val_norm, *val_anom]
        print(f' --Total Image in {self.product} Validation loader: {len(val_set)}--')

        ####  Final Data Loader ####
        self.train_loader = torch.utils.data.DataLoader(train_normal, batch_size=batch_size, shuffle=True)
        self.test_anom_loader = torch.utils.data.DataLoader(test_anom, batch_size=batch_size, shuffle=False)
        self.test_norm_loader = torch.utils.data.DataLoader(test_normal, batch_size=batch_size, shuffle=False)
        self.validation_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":

    root = '/home/roth_pu/RESIKOAST/hotspot-detection/VT-ADL/mvtech_dataset/'

    train = Mvtech(1)
    for i, j in train.test_anom_loader:
        print(i.shape)
        plt.imshow(i.squeeze(0).permute(1, 2, 0))
        plt.show
        break