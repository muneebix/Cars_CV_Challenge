#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:13:35 2019

@author: muneebix
"""

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import pandas as pd
from scipy.io import loadmat
import cv2

os.chdir(os.path.dirname(os.path.realpath(__file__)))


##preprocessing
def labels_map(PATH_TO_LABELS):
#    PATH_TO_LABELS = "./train/cars_meta.mat"
    labelsmat = loadmat(PATH_TO_LABELS)
    label = np.concatenate(labelsmat['class_names'][0]).astype(str)
    class_id = range(0,len(label))
    df = pd.DataFrame(data={"classid":class_id,"labels":label})
    df.to_csv("labels_map.csv")

labels_map("./cars/devkit/cars_meta.mat")


class Recognizer():
    def __init__(self):
        self.model = self.load_recognizer()
        self.input_shape
        self.labels = self.load_labels()
    def load_recognizer(self):
        classifier_model_path='./keras_model/inference.169-0.69.hdf5'
        classifier=load_model(classifier_model_path,compile=False)
        self.input_shape=classifier.input_shape[1:3]
        print('input shape of keras ',self.input_shape)
        return classifier
	
    def load_labels(self):
        PATH_TO_LABELS = './labels_map.csv'
        labels = pd.read_csv(PATH_TO_LABELS)
        return labels
        
    def load_images_predict(self,image_):
        #image_=image.load_img(image_,target_size=self.input_shape)
        #image_=image.img_to_array(image_)
        image_=cv2.resize(image_,(224,224))
        image_ = np.expand_dims(image_, axis=0)
        image_=image_/255.0
        #print(image_)
        res=self.model.predict(image_)
        final_label = self.labels[self.labels["classid"] == np.argmax(res)] 
            
        return final_label["labels"].values[0]
#    def inference(self):
#        load_images
#        
