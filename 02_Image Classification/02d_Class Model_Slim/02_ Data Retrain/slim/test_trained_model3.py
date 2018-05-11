# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:43:51 2018

Test the accuracy of trained model on a dataset

input:images
output:accuracy on each class and wrong classification image name

"""
from __future__ import print_function
import sys
import tensorflow as tf

tf.app.flags.DEFINE_integer('num_classes', 2, 'The number of classes.')
tf.app.flags.DEFINE_string('outfile','/home/twang/Documents/tensorflow-models/research/slim/outfile.txt', 'Output file for prediction probabilities.')
tf.app.flags.DEFINE_string('model_name', 'inception_v3', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string('preprocessing_name', 'gender','The name of the preprocessing to use. If left as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_string('checkpoint_path', '/home/twang/Documents/tensorflow-classfication2/train-slim/trained-models/airport_gender/training_all_layers','The directory where the model was written to or an absolute path to a checkpoint file.')
tf.app.flags.DEFINE_integer('eval_image_size', None, 'Eval image size.')
FLAGS = tf.app.flags.FLAGS

import numpy as np
import os
import time

from nets import nets_factory
from preprocessing import preprocessing_factory
from pandas.core.frame import DataFrame

start = time.time()

slim = tf.contrib.slim

model_name_to_variables = {'inception_v3':'InceptionV3','inception_v4':'InceptionV4','resnet_v1_50':'resnet_v1_50','resnet_v1_152':'resnet_v1_152'}

preprocessing_name = FLAGS.preprocessing_name
eval_image_size = FLAGS.eval_image_size

model_variables = model_name_to_variables.get(FLAGS.model_name)

if model_variables is None:
  tf.logging.error("Unknown model_name provided `%s`." % FLAGS.model_name)
  sys.exit(-1)

if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
  checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
else:
  checkpoint_path = FLAGS.checkpoint_path
  
# Entry to the computational graph, e.g. image_string = tf.gfile.FastGFile(image_file).read()
image_string = tf.placeholder(tf.string) 

#image = tf.image.decode_image(image_string, channels=3)
image = tf.image.decode_jpeg(image_string, channels=3, try_recover_truncated=True, acceptable_fraction=0.9) ## To process corrupted image files

image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)

network_fn = nets_factory.get_network_fn(FLAGS.model_name, FLAGS.num_classes, is_training=False)

if FLAGS.eval_image_size is None:
  eval_image_size = network_fn.default_image_size

processed_image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

processed_images  = tf.expand_dims(processed_image, 0) # Or tf.reshape(processed_image, (1, eval_image_size, eval_image_size, 3))
print("processed_images",type(processed_images),processed_images)
logits, _ = network_fn(processed_images)

probabilities = tf.nn.softmax(logits)

init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables(model_variables))

sess = tf.Session()
init_fn(sess)

fout = sys.stdout
if FLAGS.outfile is not None:
  fout = open(FLAGS.outfile, 'w')
  
h = ['image']
h.extend(['class%s' % i for i in range(FLAGS.num_classes)])
h.append('predicted_class')
print('\t'.join(h), file=fout)

output_list = []

test_img_dir = r'/media/network_shared_disk/WangTao/Person_attributes_detection/airport_gender_data/entrance'
    
test_imgs = os.listdir(test_img_dir)
size = len(test_imgs)
print("the data we test is : ",test_img_dir)
print("how many imgs in this data: ",size)

counter_female = 0
counter_male = 0

for img in test_imgs:
    
    image = os.path.join(test_img_dir,img)
    
    try:
        frame_encoded = tf.gfile.FastGFile(image,'rb').read() # You can also use x = open(fl).read()
        #print("frame_encoded",type(frame_encoded))
        image_name = os.path.basename(image)
        probs = sess.run(probabilities, feed_dict={image_string:frame_encoded})
        #np_image, network_input, probs = sess.run([image, processed_image, probabilities], feed_dict={image_string:x})
        
    except Exception as e:
        tf.logging.warn('Cannot process image file %s' % img)
        continue
 
    probs = probs[0, 0:]
    a = [image_name]
    a.extend(probs)
    a.append(np.argmax(probs))
    print('\t'.join([str(e) for e in a]), file=fout)
  
    output_list.append(a)
    
data_out = DataFrame(output_list)
data_out.to_csv('outfile_new4.csv',encoding='utf-8')
print("write outfile_new.csv to current file!")
    
end=time.time()  
    
print("everything done!")
print("the total time: ",(end-start))

sess.close()
fout.close()
