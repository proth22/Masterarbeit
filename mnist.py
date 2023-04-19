#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:30:40 2023

@author: paularoth
"""

# -*- coding: utf-8 -*-
"""
@author: Pankaj Mishra
"""
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
#import cv2
from collections import OrderedDict
from itertools import chain
import random
random.seed(123)

def read_files(root,d, product, data_motive = 'train', use_good = True, normal = True):
    '''
    return the path of the train directory and list of train images
    
    Parameters:
        root : root directory of mvtech images
        d = List of directories in the root directory
        product : name of the product to return the images for single class training.Products are-
            ['zahlen']
        data_motvie : Can be 'train' or 'test' based on the intention of the data loader function
        use_good : To use the data in the good folder. For training the default is False as we need the data of good folder.
        normal : Signofy if the normal imgaes are included while loading or not. Accepts boolean value  True or False
        
    Returns:
        Path and Image ordered dict for the test dataset
    '''
    a = os.path.join(root,d)
    print('a:', a)
    #a = os.walk(os.path.join(root,d))[1]
    #print('a:', a)
    files = next(os.walk(os.path.join(root,d)))[1]
    #walk() generates the file names in a directory tree by walking the tree either top-down or bottom-up
    #os.path.join(path, *paths) Join one or more path segments intelligently. The return value is the concatenation 
    #of path and all members of *paths, with exactly one directory separator following each non-empty part, except the last.
       
    #    print(files)
    for d_in in files:
        if os.path.isdir(os.path.join(root,d,d_in)):
            if d_in == data_motive :  #teste, ob Datei im Ordner train
                im_pt = OrderedDict()
                file = os.listdir(os.path.join(root,d, d_in))
                
    #os.path.isdir() method in Python is used to check whether the specified path is an existing directory or not
    #os.listdir method returns a list containing the names of the entries in the directory given by path
                
                for i in file:
                    if os.path.isdir(os.path.join(root, d, d_in,i)):
                        if (data_motive == 'train'):
                            tr_img_pth = os.path.join(root, d, d_in,i)
                            images = os.listdir(tr_img_pth)
                            im_pt[tr_img_pth] = images
                            print(f'total {d_in} images of {i} {d} are: {len(images)}')
                            
                        if (data_motive == 'test') :
                            if (use_good == False) and (i == 'good') and normal != True:
                                print(f'the good images for {d_in} images of {i} {d} is not included in the test anomolous data')
                            elif (use_good == False) and (i != 'good') and normal != True :
                                tr_img_pth = os.path.join(root, d, d_in,i)
                                images = os.listdir(tr_img_pth)
                                im_pt[tr_img_pth] = images
                                print(f'total {d_in} images of {i} {d} are: {len(images)}')
                            elif (use_good == True) and (i == 'good') and (normal== True):   #In diesem case sind wir
                                tr_img_pth = os.path.join(root, d, d_in,i)
                                images = os.listdir(tr_img_pth)
                                im_pt[tr_img_pth] = images
                                print(f'total {d_in} images of {i} {d} are: {len(images)}') 
                        if (data_motive == 'ground_truth'):
                            tr_img_pth = os.path.join(root, d, d_in,i)
                            images = os.listdir(tr_img_pth)
                            im_pt[tr_img_pth] = images
                            print(f'total {d_in} images of {i} {d} are: {len(images)}')
                            
                return im_pt #tr_img_pth, images
            ''' 
            Haben diesen case nicht ??
            
                if product == "all":  
                    return
                else:
                    return im_pt #tr_img_pth, images
            '''
                    
def load_images(path, image_name):
    if image_name != '.DS_Store':
        #print('path', path)
        #print('image_name',image_name)
        #array = torch.from_numpy(imread(os.path.join(path,image_name)))
        array = imread(os.path.join(path,image_name))
        #print(type(array))
        if type(array) is None :
            print('Hilfe')
        return array
    else:
        return 

    
def Test_anom_data(root, product= 'zahlen', use_good = False):
    '''
    return the path of the train directory and list of train images
    
    Parameters:
        root : root directory of mvtech images
        product : name of the product to return the images for single class training.Products are-
            ['zahlen]
        use_good : To use the data in the good folder. For training the default is False as we need the data of good folder.
        
    Returns:
        Path and Image ordered dict for the test dataset
    '''
    dir = os.listdir(root)[1]
       
    if product == 'zahlen':
             print('hier1')
             if dir == 'zahlen':
                 print('hier2')
                 pth_img_dict = read_files(root, dir, product, data_motive='test', use_good = use_good, normal = False)
                 print(pth_img_dict)
                 print('3')
                 #print('path_img:', pth_img)
                 return pth_img_dict
     
    
    '''
    for d in dir:

        if product == "all":
            read_files(root, d, product, data_motive = 'test',use_good = use_good,normal = False)
            
        elif product == d:
            pth_img_dict= read_files(root, d, product,data_motive='test', use_good = use_good, normal = False)
            return pth_img_dict
    '''
        
def Test_anom_mask(root, product= 'zahlen', use_good = False):
    '''
    return the path of the train directory and list of train images
    
    Parameters:
        root : root directory of mvtech images
        product : name of the product to return the images for single class training.Products are-
            ['zahlen']
        use_good : To use the data in the good folder. For training the default is False as we need the data of good folder.
        
    Returns:
        Path and Image ordered dict for the test dataset
    '''
    dir = os.listdir(root)[1]
    print(dir)
    
    if product == 'zahlen':
            print('hier1')
            if dir == 'zahlen':
                print('hier2')
                pth_img_dict = read_files(root, dir, product,data_motive='test', use_good = use_good, normal = False)
                print(pth_img_dict)
                print('3')
                #print('path_img:', pth_img)
                return pth_img_dict
    
    
    
    '''
    for d in dir:
        if product == "all":
            read_files(root, d, product, data_motive = 'test',use_good = use_good,normal = False)
            
        elif product == d:
            pth_img_dict= read_files(root, d, product,data_motive='ground_truth', use_good = use_good, normal = False)
            return pth_img_dict
    '''
        

def Test_normal_data(root, product= 'zahlen', use_good = True):
    '''
    if product == 'all':
        print('Please choose a valid product. Normal test data can be seen product wise')
        return
    '''
    dir = os.listdir(root)[1]
    print(dir)
    
    if product == 'zahlen':
            print('hier1')
            if dir == 'zahlen':
                print('hier2')
                pth_img = read_files(root, dir, product,data_motive='test',use_good = True, normal = True)
                print(pth_img)
                print('3')
                #print('path_img:', pth_img)
                return pth_img
    
                      
def Train_data(root, product = 'zahlen', use_good = True):
    '''
    return the path of the train directory and list of train images
    
    Parameters:
        root : root directory of mvtech images
        product : name of the product to return the images for single class training.Products are-
            ['zahlen']
        use_good : To use the data in the good folder. For training the default is True as we need the data of good folder.
        
    Returns:
        Path and Image ordered dict for the training dataset
    '''
    print('Haaalloooo')
    dir = os.listdir(root)[1]
    print('dir:', dir)
    
    #for d in dir:
    '''
        if product == "all":
            read_files(root, d, product,data_motive='train')   
        
        elif product == d:
            pth_img = read_files(root, d, product,data_motive='train')
            return pth_img
    '''
    if product == 'zahlen':
            print('hier1')
            if dir == 'zahlen':
                print('hier2')
                pth_img = read_files(root, dir, product,data_motive='train')
                print(pth_img)
                print('3')
                #print('path_img:', pth_img)
                return pth_img
        
def Process_mask(mask):
    mask = np.where(mask > 0., 1, mask)
    return torch.tensor(mask)

def ran_generator(length, shots=1):
    rand_list = random.sample(range(0,length), shots)
    return rand_list
        
        
class MNist:
   # def __init__(self, batch_size,root="/Users/paularoth/Desktop/Masterarbeit/MNISTDatasetJPGformat", product= 'zahlen'):
    def __init__(self, batch_size,root,path_train,path_test,path_anom,product= 'zahlen'):
        #The self parameter is a reference to the current instance of the class, and is used to access variables 
        #that belongs to the class
        self.root = root
        self.batch = batch_size
        self.product = product
        #self.img_ext = img_ext
        torch.manual_seed(123)
        print('1')
        # Importing all the image_path dictionaries for test and train data #
        train_path_images =Train_data(root = self.root, product = self.product)
        print('2')
        #print(train_path_images)
        print(train_path_images.keys())
        test_anom_path_images = Test_anom_data(root = self.root, product=self.product)
        print('4')
        test_anom_mask_path_images = Test_anom_mask(root = self.root, product = self.product)
        print('5')
        test_norm_path_images = Test_normal_data(root= self.root, product = self.product)
        print('6')
            
        ## Image Transformation ##
        T = transforms.Compose([
                transforms.ToPILImage(), #The Pillow library contains all the basic image processing functionality
                transforms.Resize((550,550)),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
    #            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        ''' Stack Operation zu aufwändig bei zu vielen Daten --> OutofRAM 
        1.Option:   weniger Daten
        2.Option:   Code anpassen   
        '''
        #print('train_path:', train_path_images.keys())
        #ab = load_images(train_path_images.keys()[1],train_path_images[1][1])
        #print('Hallo',train_path_images.keys())
        #print('Hallo',train_path_images)
        for i in range(1,len(train_path_images.keys())):
            for j in range(1,len(train_path_images[i])):
                print('????')
                #arra = load_images(j,i)
                #print(arra)
                
        #for i in train_path_images.keys():
            #for j in train_path_images[i]:
                #print('!!!!')
                #print(i,j)
                #arra = T(load_images(j,i))
                #print(arra)
        
        #path_train =  '/Users/paularoth/Desktop/Masterarbeit/MNISTDatasetJPGformat_small/zahlen/train/good'
        #path_test =  '/Users/paularoth/Desktop/Masterarbeit/MNISTDatasetJPGformat_small/zahlen/test/good'
        #path_anom = '/Users/paularoth/Desktop/Masterarbeit/MNISTDatasetJPGformat_small/zahlen/test/new'
        #image_name = '63.jpg'
        #pic = imread(os.path.join(path,image_name))
        #tensor = T(pic)
        #print(tensor)
        liste_train = []
        for j in train_path_images[path_train]:
            if j.endswith(".jpg"):
                #print(j)
                element = imread(os.path.join(path_train,j))
                liste_train.append(element)
        
        train_normal_image = torch.stack([T(i) for i in liste_train])
        print(train_normal_image)
                
        
        liste_test = []
        print(test_norm_path_images)
        for j in test_norm_path_images[path_test]:
            if j.endswith(".jpg"):
                #print(j)
                element = imread(os.path.join(path_test,j))
                liste_test.append(element)
        
        test_normal_image = torch.stack([T(i) for i in liste_test])
        print(test_normal_image)
        
        liste_anom = []
        for j in test_anom_path_images[path_anom]:
            if j.endswith(".jpg"):
                #print(j)
                element = imread(os.path.join(path_anom,j))
                liste_anom.append(element)
        
        test_anom_image = torch.stack([T(i) for i in liste_anom])
        print(test_anom_image)
                
         
        #.endswith(".jpg")
        #Wieso klappt nicht??
        #train_normal_image = torch.stack([T(load_images(j,i)) for j in train_path_images.keys() for i in train_path_images[j]])
        #test_anom_image = torch.stack([T(load_images(j,i)) for j in test_anom_path_images.keys() for i in test_anom_path_images[j]])
        #test_normal_image = torch.stack([T(load_images(j,i)) for j in test_norm_path_images.keys() for i in test_norm_path_images[j]])
            
        train_normal_mask = torch.zeros(train_normal_image.size(0), 1,train_normal_image.size(2), train_normal_image.size(3)  )
        test_normal_mask = torch.zeros(test_normal_image.size(0), 1,test_normal_image.size(2), test_normal_image.size(3)  )
        
        ##was gehören dort für Elemente rein??
        liste_anommask = []
        for j in test_anom_path_images[path_anom]:
            if j.endswith(".jpg"):
                #print(j)
                element = imread(os.path.join(path_anom,j))
                liste_anommask.append(element)
        
        test_anom_mask = torch.stack([T(i) for i in liste_anom])
        print(test_anom_mask)
        
        #test_anom_mask = torch.stack([Process_mask(T(load_images(j,i))) for j in test_anom_mask_path_images.keys() for i in test_anom_mask_path_images[j]])
        
        print('Bis hier 1')
        train_normal = tuple(zip(train_normal_image, train_normal_mask))
        test_anom = tuple(zip(test_anom_image, test_anom_mask))
        test_normal = tuple(zip(test_normal_image,test_normal_mask))                      
        print(f' --Size of {self.product} train loader: {train_normal_image.size()}--')
        if test_anom_image.size(0) ==test_anom_mask.size(0):
            print(f' --Size of {self.product} test anomaly loader: {test_anom_image.size()}--')
        else:
            print(f'[!Info] Size Mismatch between Anomaly images {test_anom_image.size()} and Masks {test_anom_mask.size()} Loaded')
        print(f' --Size of {self.product} test normal loader: {test_normal_image.size()}--')          
        
        print('Bis hier 2')    
        # validation set #
        num = ran_generator(len(test_anom),10)
        val_anom = [test_anom[i] for i in num]
        num = ran_generator(len(test_normal),10)
        val_norm = [test_normal[j] for j in num]
        val_set = [*val_norm, *val_anom]
        print(f' --Total Image in {self.product} Validation loader: {len(val_set)}--')
        ####  Final Data Loader ####
        self.train_loader  = torch.utils.data.DataLoader(train_normal, batch_size=batch_size, shuffle=True)            
        self.test_anom_loader = torch.utils.data.DataLoader(test_anom, batch_size = batch_size, shuffle=False)
        self.test_norm_loader = torch.utils.data.DataLoader(test_normal, batch_size=batch_size, shuffle=False)
        self.validation_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
        
        print('Bis hier 3')
            
            
if __name__ == "__main__":
    #Fuer Spyder:
    root = "/Users/paularoth/Desktop/Masterarbeit/MNISTDatasetJPGformat_small"
    #Fuer Google Colab
    #root = "/content/MNISTDatasetJPGformat"
    # print('======== All Normal Data ============')
    # Train_data(root, 'all')
    # print('======== All Anomaly Data ============')
    # Test_anom_data(root,'all')    
          
    train = MNist(1,root,'zahlen')
    for i, j in train.test_anom_loader:
        print(i.shape)
        plt.imshow(i.squeeze(0).permute(1,2,0))
        plt.show
        break
    
        
                           
                            
                
