#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:35:15 2019

@author: muneebix
"""

from architecture import CNN_model
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import os

from keras.applications.mobilenet import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping


def start(train_path,val_path):
#model=Model(inputs=mobilenet.input,outputs=prediction_layer)
    input_shape=(224,224,3)
    model=CNN_model(input_shape,196)
    #for layer in model.layers:
    #  layer.trainable=False
    #  #print(layer.name)
    #
    #for layer in model.layers[-7:]:
    #  layer.trainable=True
      #print(layer.name)
    
    data=ImageDataGenerator(preprocessing_function=preprocess_input) 
    
    train_generator=data.flow_from_directory(train_path,
                                                     target_size=(224,224),
                                                     color_mode='rgb',
                                                     batch_size=15,
                                                     class_mode='categorical',shuffle=True)
    validation_generator=data.flow_from_directory(val_path,
                                                     target_size=(224,224),
                                                     color_mode='rgb',
                                                     batch_size=15,
                                                     class_mode='categorical',shuffle=True)
    
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    os.makedirs('./keras_model_/',exist_ok=True)
    trained_models_path = './keras_model_/inference'
    
    
    
    early_stop = EarlyStopping('val_loss', patience=100)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(100/2), verbose=1)
    
    
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False)
    callbacks = [model_checkpoint, early_stop, reduce_lr]
    
    
    step_size_train=train_generator.n//train_generator.batch_size
    model.fit_generator(generator=train_generator,
                       steps_per_epoch=step_size_train,callbacks=callbacks,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.n//validation_generator.batch_size
                       ,epochs=500,verbose=1)
    