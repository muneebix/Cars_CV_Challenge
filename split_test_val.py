#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 11:54:11 2019

@author: muneebix
"""
"""
  This script splits data into validation and test folders
   A seperate folder for the validation Set is provided as split from training data is not enough for validation.
"""
import pandas as pd 
import os
import numpy as np
import shutil
import scipy.io
import train
os.chdir(os.path.dirname(os.path.realpath(__file__)))


def test_with_labels(images_path,annotation_file_path):
    images=os.listdir(images_path) 
    images=sorted(images)
    #print(images)
    
    mat = scipy.io.loadmat(annotation_file_path)
    print(mat)
    for i,image in enumerate(images):
     features=mat['annotations'][0][i]   
     class_id=features[4][0][0]
     
     name=features[5][0]
     print('class_id {} imageName {} '.format(class_id,name))
     df_=pd.DataFrame({'imagename':[name],'labels':[class_id]})
     if not os.path.isfile('./test_labels.csv'):
        df_.to_csv('./test_labels.csv', index=False)
     else: 
        df_.to_csv('./test_labels.csv', mode='a', header=False,index=False)      

def val_test_split():
    full_df=pd.read_csv('./test_labels.csv')
    print(len(full_df))
    for folder in range(1,197):
        os.makedirs('./validation_/'+str(folder),exist_ok=True)
        sub_df=full_df[full_df['labels']==folder]
        
        val_split=np.array_split(sub_df, 2)
        #print('lenth of val' ,len(val_split[0]),'lenth of test ',len(val_split[1]))
        for each in val_split[0].iterrows():
            
                name=each[1]['imagename']
                label=each[1]['labels']
                #print(name,label,' of image')a
                val_df=pd.DataFrame({'name':[name],'label':[label]})
                if not os.path.isfile('./val_set.csv'):
                
                    val_df.to_csv('./val_set.csv',index=False)
                else: 
                    val_df.to_csv('./val_set.csv', mode='a', header=False,index=False)        
        for each in val_split[1].iterrows():
                name=each[1]['imagename']
                label=each[1]['labels']
                #print(name,label,' of image')a
                test_df=pd.DataFrame({'name':[name],'label':[label]})
                if not os.path.isfile('./test_set.csv'):
                
                    test_df.to_csv('./test_set.csv',index=False)
                else: 
                    test_df.to_csv('./test_set.csv', mode='a', header=False,index=False)

    val_set=pd.read_csv('./val_set.csv')
    for row in val_set.iterrows():
        imagename=row[1]['name']
        labelname=row[1]['label']
        shutil.move('./test_/'+imagename,'./validation_/'+str(labelname))   


       
