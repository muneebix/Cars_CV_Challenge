#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:39:22 2019

@author: muneebix
"""


'''
  Preprocess file for generating train/test folder images through bboxes provided by .mat file
  just pass the path for training/testing images and training/testing annotation file  


'''


import scipy.io
import cv2
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))


def preprocess_train(images_path,annotation_file_path):
    images=os.listdir(images_path) 
    images=sorted(images)
    #print(images)
    
    mat = scipy.io.loadmat(annotation_file_path)
    
    i=0
    
    for i,image in enumerate(images):  
        
        features=mat['annotations'][0][i]
        bbx1=features[0][0][0]
        bbx2=features[1][0][0]
        bby1=features[2][0][0]
        bby2=features[3][0][0]
        class_id=features[4][0][0]
        fname=features[5][0]
        #print(bbx1,bbx2,bby1,bby2,class_id,fname)
       
        im=cv2.imread(images_path+'/'+image)
        cropped=im[bbx2:bby2,bbx1:bby1]
        #print('cropped',cropped.shape)
        os.makedirs('./train_/'+str(class_id),exist_ok=True) #  where to make folder for training
        cv2.imwrite('./train_/'+str(class_id)+'/'+fname,cropped)
        i=i+1
def preprocess_test(images_path,annotation_file_path):
    images=os.listdir(images_path) 
    images=sorted(images)
    #print(images)
    
    mat = scipy.io.loadmat(annotation_file_path)
    
    i=0
    
    for i,image in enumerate(images):  
        
        features=mat['annotations'][0][i]
        bbx1=features[0][0][0]
        bbx2=features[1][0][0]
        bby1=features[2][0][0]
        bby2=features[3][0][0]
        fname=features[4][0]
        #print(bbx1,bbx2,bby1,bby2,fname)
       
        im=cv2.imread(images_path+'/'+image)
        cropped=im[bbx2:bby2,bbx1:bby1]
        #print('cropped',cropped.shape)
        os.makedirs('./test_/',exist_ok=True) #  where to make folder for training
        cv2.imwrite('./test_/'+fname,cropped)
        i=i+1
        
           
           
          