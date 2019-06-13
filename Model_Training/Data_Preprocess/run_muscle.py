# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:07:58 2019

@author: YannanLin
"""

import utils_muscle
import glob
import numpy as np
import os
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

################################# set paths ###################################

mlo_image_path = '/home/yannan_lin/unet/MLO_image/'
mlo_mask_path = '/home/yannan_lin/unet/MLO_mask/'

test_image_path = '/home/yannan_lin/unet/test_image/'
test_gt_path = '/home/yannan_lin/unet/test_gt/'
test_result_path = '/home/yannan_lin/unet/test_result/'

model_path = '/home/yannan_lin/unet/'

image_path = glob.glob("%s/*.jpg" % mlo_image_path)
mask_path = glob.glob("%s/*.jpg" % mlo_mask_path)

test_path = glob.glob("%s/*.jpg" % test_image_path)
truth_path = glob.glob("%s/*.jpg" % test_gt_path)
result_path = glob.glob("%s/*.jpg" % test_result_path)

side_length = 256

################################## training ###################################

# create train data
image_list, binary_mask_list, name_list = create_train_data(image_path, mask_path, side_length)

#rotate images
img_list_rotate = rotate_images(image_list, binary_mask_list)[0]
mask_list_rotate = rotate_images(image_list, binary_mask_list)[1]

#flip images
img_list_fliplr = flip(image_list, binary_mask_list)[0]
mask_list_fliplr = flip(image_list, binary_mask_list)[1]
img_list_flipud = flip(image_list, binary_mask_list)[2]
mask_list_flipud = flip(image_list, binary_mask_list)[3]
        
#concatenate img and mask lists
img_list_whole = image_list+img_list_rotate+img_list_fliplr+img_list_flipud
mask_list_whole = binary_mask_list+mask_list_rotate+mask_list_fliplr+mask_list_flipud

# split train and test data
x_train, x_test, y_train, y_test = train_test_split(img_list_whole, mask_list_whole, test_size=0.2, random_state=42)

# resize train and test data
x_train = np.asarray([x_train[i].reshape([side_length,side_length,1]) for i in range(len(x_train))])
x_test = np.asarray([x_test[i].reshape([side_length,side_length,1]) for i in range(len(x_test))])

y_train = np.asarray([y_train[i].reshape([side_length,side_length,1]) for i in range(len(y_train))])
y_test = np.asarray([y_test[i].reshape([side_length,side_length,1]) for i in range(len(y_test))])

# compile and train model
model = unet(input_size = (side_length,side_length,1))

callbacks = [
EarlyStopping(patience=20, verbose=1),
ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
ModelCheckpoint(os.path.join(model_path+'model.h5'), verbose=1, save_best_only=True)
]

# train model
history = model.fit(x_train, y_train, batch_size=16, epochs=300, callbacks=callbacks,
validation_data=(x_test, y_test))

# plot tranin and validation loss & acc
plot_acc_loss(history)

################################### testing ###################################

# read test data
test_image_list, test_gt_list = read_test_images(test_path, truth_path, side_length)

# test model using test images
test_model(test_path, test_image_path, test_image_list, 
               test_result_path, side_length,model_path)

# print test results
print_test_result(40,0.7,model_path,
                  test_image_list, test_gt_list,
                  side_length) # 0 is index, 0.7 is threshold

################################# evaluation ##################################

# read test results
test_result_list = read_test_results(result_path, side_length)

# print jaccard index
iou_list, mean_iou = mean_iou(test_gt_list, test_result_list)
print(iou_list)
print(mean_iou)

########################## post procesing #####################################

bad_image_index_list = [5,9,11,14,15,18,29,44,49,17,18,20,21,30,47]
test_result_list, new_result_list = post_processing(result_path, side_length)

plot_post_process_results(test_result_list, new_result_list,49)
