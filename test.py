# -*- coding: utf-8 -*-
"""
@author: Pankaj Mishra
"""

import torch
import mnist2
import mnist3
import torch.nn.functional as F
import os
import numpy as np
import pytorch_ssim
from torchmetrics.functional import structural_similarity_index_measure as sim_loss
from einops import rearrange
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
#import mdn1
from VT_AE import VT_AE as ae
from utility_fun import *
import matplotlib.pyplot as plt

patch_size = 4

#ssim_loss = pytorch_ssim.SSIM() # SSIM Loss

# Dataset
data_MNist = mnist3.MNist(1)
image_size=28
print('MNIST data has been successfully imported')

# Model declaration
model = ae(train=False, image_size=image_size, patch_size=patch_size).cuda()
#G_estimate= mdn1.MDN().cuda()

# Loading weights
model.load_state_dict(torch.load(f'./saved_model/VT_AE_MNist'+'.pt'))
#G_estimate.load_state_dict(torch.load(f'./saved_model/G_estimate_MNist_{prdt}'+'.pt'))

#put model to eval
model.eval()
#G_estimate.eval()


#### testing #####
loader = [data_MNist.train_loader,data_MNist.test_norm_loader,data_MNist.test_anom_loader]

t_loss_norm =[]
t_loss_anom =[]

def Thresholding(data_load = loader[1:], upsample = 0, thres_type = 0, fpr_thres = 0.3):
    '''
    Parameters
    ----------
    data : TYPE, optional
        DESCRIPTION. The default is data.train_loader.
    upsample : INT, optional
        DESCRIPTION. 0 - NearestUpsample2d; 1- BilinearUpsampling.
    thres_type : INT, optional
        DESCRIPTION. 0 - 30% of fpr reached; 1 - thresholding using best F1 score
    fpr_thres : FLOAT, Optional
        DESCRIPTION. False Positive Rate threshold value. Default is 0.3

    Returns
    -------
    Threshold: Threshold value

    '''
    norm_loss_t = []
    normalised_score_t = []
    mask_score_t = []

    for data in data_load:
        for i, j in data:
            if i.size(1)==1:
                i = torch.stack([i,i,i]).squeeze(2).permute(1,0,2,3)
            vector, reconstructions = model(i.cuda())
            #pi, mu, sigma = G_estimate(vector)
            
            #Loss calculations
            loss1 = F.mse_loss(reconstructions,i.cuda(), reduction='mean') #Rec Loss
            loss2 = -sim_loss(i.cuda(), reconstructions) #SSIM loss for structural similarity
            #loss3 = mdn1.mdn_loss_function(vector,mu,sigma,pi, test= True) #MDN loss for gaussian approximation
            #loss = loss1 + loss2 + loss3.sum()       #Total loss
            loss = loss1
            norm_loss_t.append(loss.detach().cpu().numpy())
                
            if upsample==0 :
                #Mask patch
                mask_patch = rearrange(j.squeeze(0).squeeze(0), '(h p1) (w p2) -> (h w) p1 p2', p1 = patch_size, p2 = patch_size)
                mask_patch_score = Binarization(mask_patch.sum(1).sum(1),0.)
                mask_score_t.append(max(mask_patch_score)) # Storing all masks
                norm_score = norm_loss_t[-1]
                normalised_score_t.append(norm_score)# Storing all patch scores
            elif upsample == 1:
                mask_score_t.append(j.squeeze(0).squeeze(0).cpu().numpy()) # Storing all masks
                m = torch.nn.UpsamplingBilinear2d((512,512))
                norm_score = norm_loss_t[-1]
                #norm_score = norm_loss_t[-1].reshape(-1,1,512//patch_size,512//patch_size)
                score_map = m(torch.tensor(norm_score))
                score_map = Filter(score_map , type =0) # add normalization here for the testing
                normalised_score_t.append(score_map) # Storing all score maps
                
                
    scores = np.asarray(normalised_score_t).flatten()
    masks = np.asarray(mask_score_t).flatten()
    
    if thres_type == 0 :
        fpr, tpr, _ = roc_curve(masks, scores)
        ROC = [fpr, tpr, _]
        precision, recall, thresholds = precision_recall_curve(masks, scores)
        PRC = [precision, recall, thresholds]
        fp3 = np.where(fpr <= fpr_thres)
        threshold = _[fp3[-1][-1]]

        fig, ax = plt.subplots()
        ax.plot(_[1:],fpr[1:])
        ax.set_xlabel('Thresholds')
        ax.set_ylabel('false positiv ratio')
        plt.show()
        fig.savefig(f'Threshold_to_fpr')
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(thresholds, precision[1:]) #np.insert(thresholds, 0, 0.0)
        ax.set_xlabel('Thresholds')
        ax.set_ylabel('Precision')
        plt.show()
        fig.savefig(f'Threshold_to_precision')
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(thresholds, recall[1:]) #np.insert(thresholds, 0, 0.0)
        ax.set_xlabel('Thresholds')
        ax.set_ylabel('Recall')
        plt.show()
        fig.savefig(f'Threshold_to_recall')
        plt.close(fig)

    elif thres_type == 1:
        precision, recall, thresholds = precision_recall_curve(masks, scores)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)] 
    return threshold, ROC, PRC
    

def Patch_Overlap_Score(threshold, data_load = loader[1:], upsample =1):
    '''
    Parameters
    ----------
    threshold: FLOAT, optional
        DESCRIPTION: threshold determined in Thresholding()
    data : TYPE, optional
        DESCRIPTION. The default is data.train_loader.
    upsample : INT, optional
        DESCRIPTION. 0 - NearestUpsample2d; 1- BilinearUpsampling.

    Returns
    -------
    PRO_score:
        DESCRIPTION:
    AUC_Score_total:
        DESCRIPTION:
    AUC_PR:
        DESCRIPTION:

    '''
    norm_loss_t = []
    normalised_score_t = []
    mask_score_t = []
    loss1_tn = []
    loss2_tn = []
    loss_tn = []
    #loss3_tn = []
    loss1_ta = []
    loss2_ta = []
    loss_ta = []
    #loss3_ta = []
    
    score_tn = []
    score_ta = []
    
    counter_norm = 0
    counter_anom = 0
    for n,data in enumerate(data_load):
        total_loss_all = []
        for c,(i, j) in enumerate(data):
            if i.size(1)==1:
                i = torch.stack([i,i,i]).squeeze(2).permute(1,0,2,3)
            vector, reconstructions = model(i.cuda())
            #pi, mu, sigma = G_estimate(vector)

            #Plotting anom reconstructions
            if n == 1 and counter_anom < 30:
                # Plot of reconstructions
                tensor1 = reconstructions.squeeze()  # Entfernen Sie die erste Dimension mit Länge 1
                tensor1 = np.transpose(tensor1.cpu().detach().numpy(),(1, 2, 0))  # Ändern Sie die Reihenfolge der Achsen
                tensor1 = tensor1.mean(axis=2)
                # Plot of Image
                tensor2 = i.squeeze()
                tensor2 = np.transpose(tensor2.cpu().detach().numpy(),(1, 2, 0))  # Ändern Sie die Reihenfolge der Achsen
                tensor2 = tensor2.mean(axis=2)
                # Plot erstellen
                counter_anom += 1
                fig, axs = plt.subplots(1, 2)
                imag = axs[0].imshow(tensor1, cmap='gray')
                axs[0].axis('off')
                axs[0].set_title('Reconstructed Image')

                axs[1].imshow(tensor2, cmap='gray')
                axs[1].axis('off')
                axs[1].set_title('Original Image')
                plt.show()

                #fig.colorbar(imag, cmap='gray')  # Add a colorbar with the desired colormap
                fig.savefig(f'plot_{counter_anom}_recons_anom.png')

                plt.close(fig)

            # Plotting norm reconstructions
            elif n == 0 and counter_norm < 10:
                # Plot of reconstructions
                tensor1 = reconstructions.squeeze()  # Entfernen Sie die erste Dimension mit Länge 1
                tensor1 = np.transpose(tensor1.cpu().detach().numpy(), (1, 2, 0))  # Ändern Sie die Reihenfolge der Achsen
                tensor1 = tensor1.mean(axis=2)
                # Plot of Image
                tensor2 = i.squeeze()
                tensor2 = np.transpose(tensor2.cpu().detach().numpy(),(1, 2, 0))  # Ändern Sie die Reihenfolge der Achsen
                tensor2 = tensor2.mean(axis=2)
                # Plot erstellen
                counter_norm += 1
                fig, axs = plt.subplots(1,2)
                imag = axs[0].imshow(tensor1, cmap='gray')
                axs[0].axis('off')
                axs[0].set_title('Reconstructed Image')

                axs[1].imshow(tensor2, cmap='gray')
                axs[1].axis('off')
                axs[1].set_title('Original Image')
                plt.show()

                #fig.colorbar(imag, cmap='gray')  # Add a colorbar with the desired colormap
                fig.savefig(f'plot_{counter_norm}_recons_norm.png')

                plt.close(fig)

            #Loss calculations
            loss1 = F.mse_loss(reconstructions,i.cuda(), reduction='mean') #Rec Loss
            loss2 = -sim_loss(i.cuda(), reconstructions) #SSIM loss for structural similarity
            #loss3 = mdn1.mdn_loss_function(vector,mu,sigma,pi, test= True) #MDN loss for gaussian approximation
            #loss = loss1 -loss2 + loss3.max()       #Total loss
            loss = loss1
            #norm_loss_t.append(loss3.detach().cpu().numpy())
            total_loss_all.append(loss.detach().cpu().numpy())
            
            if n == 0 : #data = data.test_norm_loader
                loss1_tn.append(loss1.detach().cpu().numpy())
                loss2_tn.append(loss2.detach().cpu().numpy())
                loss_tn.append(loss.detach().cpu().numpy())
                #loss3_tn.append(loss3.sum().detach().cpu().numpy())
            if n == 1:  #data = data.test_anom_loader
                loss1_ta.append(loss1.detach().cpu().numpy())
                loss2_ta.append(loss2.detach().cpu().numpy())
                loss_ta.append(loss.detach().cpu().numpy())
                #loss3_ta.append(loss3.sum().detach().cpu().numpy())
            '''  
            if upsample==0 :
                #Mask patch
                mask_patch = rearrange(j.squeeze(0).squeeze(0), '(h p1) (w p2) -> (h w) p1 p2', p1 = patch_size, p2 = patch_size)
                mask_patch_score = Binarization(mask_patch.sum(1).sum(1),0.)
                mask_score_t.append(mask_patch_score) # Storing all masks
                norm_score = Binarization(norm_loss_t[-1], threshold)
                m = torch.nn.UpsamplingNearest2d((512,512))
                score_map = m(torch.tensor(norm_score.reshape(-1,1,512//patch_size,512//patch_size)))     
                
                normalised_score_t.append(norm_score)# Storing all patch scores

            elif upsample == 1:
                mask_score_t.append(j.squeeze(0).squeeze(0).cpu().numpy()) # Storing all masks
                
                m = torch.nn.UpsamplingBilinear2d((512,512))
                norm_score = norm_loss_t[-1].reshape(-1,1,512//patch_size,512//patch_size)
                score_map = m(torch.tensor(norm_score))
                score_map = Filter(score_map , type =0)

                   
                normalised_score_t.append(score_map) # Storing all score maps

            ## Plotting
            if c%5 == 0:
                plot(i,j,score_map[0][0])
            if n == 0:
                score_tn.append(score_map.max())
            if n ==1:
                score_ta.append(score_map.max())
            '''

        if n == 0 :
            t_loss_all_normal = total_loss_all
        if n == 1:
            t_loss_all_anomaly = total_loss_all
    '''
    ## PRO Score            
    scores = np.asarray(normalised_score_t).flatten()
    masks = np.asarray(mask_score_t).flatten()
    PRO_score = roc_auc_score(masks, scores)
    '''
    ## Image Anomaly Classification Score (AUC)
    roc_data = np.concatenate((t_loss_all_normal, t_loss_all_anomaly))
    roc_targets = np.concatenate((np.zeros(len(t_loss_all_normal)), np.ones(len(t_loss_all_anomaly))))
    AUC_Score_total = roc_auc_score(roc_targets, roc_data)
    
    # AUC Precision Recall Curve
    precision, recall, thres = precision_recall_curve(roc_targets, roc_data)
    AUC_PR = auc(recall, precision)

    
    return AUC_Score_total, AUC_PR, loss1_ta, loss1_tn, loss2_ta, loss2_tn, loss_ta, loss_tn

if __name__=="__main__":
    thres, ROC, PRC = Thresholding()
    print(f'Threshold: {thres}')
    #print(ROC)
    #PRO, AUC, AUC_PR = Patch_Overlap_Score(threshold=thres)
    AUC, AUC_PR, loss1_ta, loss1_tn, loss2_ta, loss2_tn, loss_ta, loss_tn = Patch_Overlap_Score(threshold=thres)

    #print(f'PRO Score: {PRO} \nAUC Total: {AUC} \nPR_AUC Total: {AUC_PR}')
    print(f'AUC Total: {AUC} \nAUC_PR Total: {AUC_PR}')

    print(f'Loss_ta mean: {np.mean(loss_ta)}, Loss_ta_std: {np.std(loss_ta)}')
    print(f'Loss_tn mean: {np.mean(loss_tn)}, Loss_tn_std: {np.std(loss_tn)}')

    print(f'Loss1_ta mean: {np.mean(loss1_ta)}, Loss1_ta_std: {np.std(loss1_ta)}')
    print(f'Loss1_tn mean: {np.mean(loss1_tn)}, Loss1_tn_std: {np.std(loss1_tn)}')

    print(f'Loss2_ta mean: {np.mean(loss2_ta)}, Loss2_ta_std: {np.std(loss2_ta)}')
    print(f'Loss2_tn mean: {np.mean(loss2_tn)}, Loss2_tn_std: {np.std(loss2_tn)}')

    fig, axs = plt.subplots(1, 2)
    # Erzeuge ein Histogramm
    axs[0].hist(loss1_ta, bins=30)  # Anzahl der Bins anpassen
    axs[0].set_title('loss1_ta Histogramm')

    axs[1].hist(loss1_tn, bins=30)  # Anzahl der Bins anpassen
    axs[1].set_title('loss1_tn Histogramm')
    plt.show()
    fig.savefig(f'histogramm_loss1_ta_tn.png')
    plt.close(fig)

    if loss_ta != loss1_ta:
        fig, axs = plt.subplots(1, 2)
        # Erzeuge ein Histogramm
        axs[0].hist(loss_ta, bins=30)  # Anzahl der Bins anpassen
        axs[0].set_title('loss_ta Histogramm')

        axs[1].hist(loss_tn, bins=30)  # Anzahl der Bins anpassen
        axs[1].set_title('loss_tn Histogramm')
        plt.show()
        fig.savefig(f'histogramm_loss_ta_tn.png')
        plt.close(fig)

