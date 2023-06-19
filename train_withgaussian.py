# -*- coding: utf-8 -*-
"""
@author: Pankaj Mishra
"""

import torch
import mnist2
import mnist3
import torchvision.utils as utils
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import structural_similarity_index_measure as sim_loss
import os
import numpy as np
import pytorch_ssim
import mdn1
from VT_AE import VT_AE as ae
import argparse

## Argparse declaration ##

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", required=False, default=50, help="Number of epochs to train")
ap.add_argument("-lr", "--learning_rate", required=False, default=0.0001, help="learning rate")
ap.add_argument("-ps", "--patch_size", required=False, default=4, help="Patch size of the images")
ap.add_argument("-b", "--batch_size", required=False, default=8, help="batch size")
args = vars(ap.parse_args())

writer = SummaryWriter()

epoch = args["epochs"]
minloss = 1e10
ep = 0

loss_weighting = [30,1,0.001]  # weighting of [loss1,loss2,loss3]

# SSIM Loss (Image spezefisch, misst die strukturelle Bildähnlichkeit -> measure of the similarity by comparing two
# images based on luminance similarity, contrast similarity and structural similarity information
#ssim_loss = pytorch_ssim.SSIM()

# Dataset
data = mnist3.MNist(args["batch_size"])
image_size = 28
print('Mnist data has been successfully imported')

# Model declaration
model = ae(patch_size=args["patch_size"], train=True).cuda()
G_estimate = mdn1.MDN().cuda()

### put model to train ##
# (The two models are trained as a separate module so that it would be easy to use as an independent module in different scenarios)
model.train()
G_estimate.train()

# Optimiser Declaration
Optimiser = Adam(list(model.parameters()) + list(G_estimate.parameters()), lr=args["learning_rate"], weight_decay=0.0001)

############## TRAIN #####################
# torch.autograd.set_detect_anomaly(True) #uncomment if you want to track an error

loss2_array_mean = []
loss1_array_mean = []
loss3_array_mean = []

print('\nNetwork training started.....')
for i in range(epoch):
    t_loss = []
    loss2_array = []
    loss1_array = []
    loss3_array = []

    for j, m in data.train_loader:
        if j.size(1) == 1:
            j = torch.stack([j, j, j]).squeeze(2).permute(1, 0, 2, 3)
        model.zero_grad()

        # vector,pi, mu, sigma, reconstructions = model(j.cuda())
        vector, reconstructions = model(j.cuda())
        pi, mu, sigma = G_estimate(vector)

        # Loss calculations
        loss1 = F.mse_loss(reconstructions, j.cuda(), reduction='mean')  # Rec Loss
        loss2 = -sim_loss(j.cuda(), reconstructions)  # SSIM loss for structural similarity
        loss3 = mdn1.mdn_loss_function(vector, mu, sigma, pi)  # MDN loss for gaussian approximation
        loss1_array.append(loss1.item())
        loss2_array.append(loss2.item())
        loss3_array.append(loss3.item())

        #print(f' loss3  : {loss3.item()}')
        #loss = loss1 # Total loss
        loss = loss_weighting[0] * loss1 + loss_weighting[1] * loss2 + loss_weighting[2] * loss3  # Total loss

        t_loss.append(loss.item())  # storing all batch losses to calculate mean epoch loss

        # Tensorboard definitions
        writer.add_scalar('recon-loss', loss1.item(), i)
        writer.add_scalar('ssim loss', loss2.item(), i)
        writer.add_scalar('Gaussian loss', loss3.item(), i)
        writer.add_histogram('Vectors', vector)

        ## Uncomment below to store the distributions of pi, var and mean ##
        # writer.add_histogram('Pi', pi)
        # writer.add_histogram('Variance', sigma)
        # writer.add_histogram('Mean', mu)

        # Optimiser step
        loss.backward()
        Optimiser.step()

    # Tensorboard definitions for the mean epoch values
    writer.add_image('Reconstructed Image', utils.make_grid(reconstructions), i, dataformats='CHW')
    writer.add_scalar('Mean Epoch loss', np.mean(t_loss), i)
    print(f'Mean Epoch {i} loss: {np.mean(t_loss)}')
    print(f'Min loss epoch: {ep} with min loss: {minloss}')

    writer.close()
    loss1_array_mean.append(np.mean(loss1_array))
    loss2_array_mean.append(np.mean(loss2_array))
    loss3_array_mean.append(np.mean(loss3_array))
    print(f'loss_1 epoch: {np.mean(loss1_array)} \nloss_2 epoch:: {np.mean(loss2_array)} \nloss_3 epoch: {np.mean(loss3_array)}')

    # Saving the best model
    if np.mean(t_loss) <= minloss:
        minloss = np.mean(t_loss)
        ep = i
        os.makedirs('./saved_model', exist_ok=True)
        torch.save(model.state_dict(), f'./saved_model/VT_AE_MNist_withgaussian' + '.pt')
        torch.save(G_estimate.state_dict(), f'./saved_model/G_estimate_MNist_withgaussian' + '.pt')

'''
Full forms:
GN - gaussian Noise
LD = Linear Decoder
DR - Dynamic Routing
Gn = No of gaussian for the estimation of density, with n as the number
Pn = Patch with n is dim of patch
SS - trained with ssim loss
'''