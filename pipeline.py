#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:36:29 2019

"""
import argparse
import recognizer as rec
import detector as det
import os
from PIL import Image
from keras.preprocessing import image
import preprocess
import split_test_val
import train
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", required=True,
	help="Select a mode for operations; 1. train 2. test")
ap.add_argument("-d", "--data", required=False,
	help="data directory")
args = vars(ap.parse_args())
train_path='./train_'
val_path='./val_'
os.chdir(os.path.dirname(os.path.realpath(__file__)))
def run(args):
    if args['mode'] == 'train':
        ## preprocessing steps
        preprocess.preprocess_train('./cars/cars_train','./cars/devkit/cars_train_annos.mat')
        preprocess.preprocess_test('./cars/cars_test','./cars/devkit/cars_test_annos.mat')
        
        split_test_val.test_with_labels('./cars/cars_test','./cars/devkit/cars_test_annos_withlabels.mat')
        split_test_val.val_test_split()
        
        
        train.start(train_path,val_path)
        
        
    elif args['mode'] == 'test':
        car_detector = det.Detector()
        car_recognizer = rec.Recognizer() 
        images_dir = os.listdir(args['data']+"/test/")   
        for imagepath  in images_dir:
            no_of_cars,car_boxes = car_detector.test_model(args['data']+"/test/"+imagepath)
            print('car_boxes',car_boxes[0])
            if no_of_cars > 0:
                print(imagepath)
                image_=image.load_img(args['data']+"/test/"+imagepath)
                image_=image.img_to_array(image_)
                height,width,ch = image_.shape
                if no_of_cars > 1:
                    for cars in range(0,no_of_cars):
                        print('cars',cars)
                        rec_image = image_[int(car_boxes[cars][0]*height):int(car_boxes[cars][2]*height),int(car_boxes[cars][1]*width):int(car_boxes[cars][3]*width)]
                        result = car_recognizer.load_images_predict(rec_image)
                        
                        print("found {} in the above image".format(result))
                else:
                    rec_image = image_[int(car_boxes[0][0]*height):int(car_boxes[0][2]*height),int(car_boxes[0][1]*width):int(car_boxes[0][3]*width)]
                    result = car_recognizer.load_images_predict(rec_image)
                    print("found {} in the above image".format(result))    
                    
                    
                
run(args)            
            
        
        
    
    
