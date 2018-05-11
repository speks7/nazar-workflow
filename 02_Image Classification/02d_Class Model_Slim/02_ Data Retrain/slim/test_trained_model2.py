# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 16:47:01 2018

@author: twang

Test the accuracy of trained model on a dataset

input:images
output:accuracy on each class and wrong classification image name

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import cv2

import numpy as np
import tensorflow as tf

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph



def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label
  
def prediction(graph,test_img_path,test_imgs):
    
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer="input"
    output_layer="InceptionV3/Predictions/Reshape_1"
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    
    size = len(test_imgs)
    
    counter_female = 0
    counter_male = 0
    
    print("the data we test is : ",test_img_path)
    print("how many imgs in this data: ",size)
    log_list_female = []
    log_list_male = []
    
    for img in test_imgs:
        
        image = os.path.join(test_img_path,img)
        
        try:
            frame = cv2.imread(image)
            resized = cv2.resize(frame,(input_height,input_width))
            float_img=np.asfarray(resized,dtype="float32")
            image_np_expanded = np.expand_dims(float_img,axis=0)
            normalized = np.divide(np.subtract(image_np_expanded,[input_mean]),[input_std])
        except:
            continue
            print("this image we don't find: ",image)
        

      
        with tf.Session(graph=graph) as sess:
          
            # Feed the image_data as input to the graph and get first prediction
            input_operation = graph.get_operation_by_name(input_name)
         
            output_operation = graph.get_operation_by_name(output_name)
         
            results = sess.run(output_operation.outputs[0],{input_operation.outputs[0]:normalized})
         
        results1 = np.squeeze(results) 
      
        if results1[0]>results1[1]:
            counter_female +=1
            log_list_female.append(image)
            if counter_female%20==0:
                print("counter_female",counter_female)
            else:
                pass
        else :
            counter_male +=1
            log_list_male.append(image)
            if counter_male%20==0:
                print("counter_male",counter_male)
            else:
                pass
    
    female = open("file_log_female.txt","w")
    female.write('\n'.join(log_list_female))
    female.close()
    print("write the file_log_female.txt")

    male = open("file_log_male.txt","w")
    male.write('\n'.join(log_list_male))
    male.close()
    print("write the file_log_male.txt")
    
    print("the female classfication is: ",counter_female/size)  
    print("the male classfication is :",counter_male/size)                      

if __name__ == "__main__":
    
    start = time.time()
    
    MODEL_DIR = r"/media/network_shared_disk/WangTao/Person_attributes_detection/COCOPersonDetection+RAPBodyPartition+RAPGender/tensorlow_models"
    
    Gender_Classify_MODEL = os.path.join(MODEL_DIR,"gender_inception_all_layers_100000")
    Gender_Classify_CKPT = os.path.join(Gender_Classify_MODEL,'frozen_inception_v3.pb')
    Gender_Classify_LABELS = os.path.join(Gender_Classify_MODEL,'labels.txt')
  
    graph = load_graph(Gender_Classify_CKPT)
  
    test_img_dir = r'/media/network_shared_disk/WangTao/Person_attributes_detection/gender_data/validation/all'
    
    test_imgs = os.listdir(test_img_dir)
    
    prediction(graph,test_img_dir,test_imgs)
    
    #prediction(graph,test_img_path_male, test_imgs_male)
    
    end=time.time()  
    
    print("everything done!")
    print("the total time: ",(end-start))
               