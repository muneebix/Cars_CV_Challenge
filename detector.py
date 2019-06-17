#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 21:47:58 2019

@author: ix
"""

import numpy as np
import os

import sys
import cv2
import tensorflow as tf
#from keras.preprocessing import load_image
from PIL import Image
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("..")
#from utils import ops as utils_ops


from utils import label_map_util

class Detector():
    def __init__(self):
        self.model = self.load_detector()
        
    def load_detector(self):
        # What model to download.
        MODEL_NAME = './ssdlite'
        
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
        
        
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')              
        return detection_graph

    def load_labels():
        PATH_TO_LABELS = './ssdlite/mscoco_label_map.pbtxt'
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        return category_index

    def load_image_into_numpy_array(self,image):
      (im_width, im_height) = image.size
      return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)

    def run_inference_for_single_image(self,image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                #model = (img_size,stage_num, lambda_local, lambda_d)()
              # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
#                
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    
                # Run inference
                output_dict = sess.run(tensor_dict,
                                     feed_dict={image_tensor: image})
    
                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
#                if 'detection_masks' in output_dict:
#                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

  
  
    def test_model_multiple(self,PATH_TO_TEST_IMAGES_DIR):
        #PATH_TO_TEST_IMAGES_DIR = 'test_images'
        TEST_IMAGE_PATHS = os.listdir(PATH_TO_TEST_IMAGES_DIR)
        # Size, in inches, of the output images.
        for image_path in TEST_IMAGE_PATHS:
            print(image_path)
            image = Image.open(os.path.join(PATH_TO_TEST_IMAGES_DIR,image_path))
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = self.load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            output_dict = self.run_inference_for_single_image(image_np_expanded, self.model)
            classes = output_dict["detection_classes"]
            scores = output_dict["detection_scores"]
            boxes = output_dict["detection_boxes"]
            indices = np.where(classes==3)
            indices = np.where(scores[indices]>=0.5)
            final_boxes = boxes[indices]
        return len(final_boxes),final_boxes

    def test_model(self,PATH_TO_TEST_IMAGE):
        #PATH_TO_TEST_IMAGES_DIR = 'test_images'
        print('pth is ',PATH_TO_TEST_IMAGE)
        image = Image.open(PATH_TO_TEST_IMAGE)
        
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = self.load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = self.run_inference_for_single_image(image_np_expanded, self.model)
        classes = output_dict["detection_classes"]
        scores = output_dict["detection_scores"]
        boxes = output_dict["detection_boxes"]
        indices = np.where(classes==3)
        indices = np.where(scores[indices]>=0.5)
        final_boxes = boxes[indices]
        return len(final_boxes),final_boxes

    
    
#    detector = Detector()
#    image_='./test.jpg'
#    no_of_cars,bboxes = detector.test_model(image_)
#    image_=image.load_img(image_,target_size=(224, 224))
#    
#    detector
#    
#    
#    
#    #classifier=
#    h,w,ch=im.shape
#    for i,bb in enumerate(bboxes):
#        
#        ymin=bb[0]*h
#        xmin=bb[1]*w
#        ymax=bb[2]*h
#        xmax=bb[3]*w
#        im=cv2.rectangle(im,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),3)
#        print(ymin,ymax,xmin,xmax)
##    
#    cv2.imwrite('/home/muneebix/aiforsea/result.jpg',im)
#    print("There are {} number of cars".format(no_of_cars))
#    print("these are the bounding boxes for them {}".format(bboxes))
#    
#main()
     
