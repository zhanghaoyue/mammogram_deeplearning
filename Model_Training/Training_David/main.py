############################Import Packages#####################################

from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

import os
import re
import io
import csv
import cv2
import random
import sys
import glob
import re
import imageio
import itertools

#%load_ext autoreload
#%autoreload 2
#%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from scipy import ndarray, ndimage
from skimage import io, color, transform, util, morphology, measure, filters
from skimage.color import rgb2gray
from skimage.io import imsave, imshow, imread
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imshow, imsave
from skimage.color import rgb2gray, label2rgb
from skimage.segmentation import clear_border
from skimage.morphology import ball, disk, remove_small_objects
from skimage.morphology import erosion, dilation, closing, reconstruction, square
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening
from skimage.measure import label, regionprops, perimeter
from skimage.filters import roberts, sobel, threshold_otsu
import matplotlib.patches as mpatches
from sklearn.feature_extraction import image
import get_dataset
import model_attn_mil
import mg_bag_loader

import SimpleITK as sitk

"""
Implementing the Attention-based Multi Instance Learning Model Using Mammograms.

:epochs: number of epochs to train model.
:learning rate: learning rate to train model.
:weight decay: weight decay to train model.
:target number: the desired bag label number. In this project, use 1, as bags have a positive label (1)
if they contain at least 1 cancerous patch or a negative label (0) if they do not contain any cancerous patches.
The aim of our model is to predict this target number (the bag label).
:number_of_patches: the desired number of patches.  In this project, we extracted 50 patches of size 128 x 128
from the image after it was segmented.  These number of patches (50) are contained within one bag.
:variance_of_number_patches: the desired variance for the number of patches.  In this project, we set to 0
because we have a fixed number of patches extracted from the image which are contained within one bag.
:num_bags_train: This contains the number of bags in the training model, which can also be interpreted as
the number of images in your training set before having extracted patches.  In this project, we set to 522.
:num_bags_test: This contains the number of bags in the test model, which can also be interpreted as
the number of images in your test set before having extracted patches.  In this project, we set to 58.
:seed: set random seed.
:no-cuda: disables CUDA training.

Output:
:predicted label:  The predicted label for bag (cancer or no-cancer).
:error: the error.

"""


#########################Training settings######################################

parser = argparse.ArgumentParser(description='PyTorch MG Bags Implementation')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)') #could change to 10
parser.add_argument('--lr', type=float, default=.0005, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=.00005, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=1, metavar='T',
                    help='bags have a positive label if they contain at least one 1') #might be able to leave
parser.add_argument('--number_of_patches', type=int, default=50, metavar='ML',
                    help='number of patches in bag') # number of patches (more than about 60 patches produces cannot allocate memory error)
parser.add_argument('--variance_of_number_patches', type=int, default=0, metavar='VL',
                    help='variance of number of patches') # do not need variance, because have fixed number of patches
parser.add_argument('--num_bags_train', type=int, default=522, metavar='NTrain',
                    help='number of bags in training set') # number of original images
parser.add_argument('--num_bags_test', type=int, default=58, metavar='NTest',
                    help='number of bags in test set') # number of original images
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

print('Initialize Model')
model = Attention()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)


def to_np(v):
    """
    Returns an np.array object given an input of np.array, list, tuple, torch variable or tensor.
    Needed to get the train loss and the test loss.
    """
    if isinstance(v, float): return np.array(v)
    if isinstance(v, (np.ndarray, np.generic)): return v
    if isinstance(v, (list,tuple)): return [to_np(o) for o in v]
    if isinstance(v, Variable): v=v.data
    if torch.cuda.is_available() and is_half_tensor(v): v=v.float()
    if isinstance(v, torch.FloatTensor): v=v.float()
    return v.cpu().numpy()

def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader): #train loader should be of length: number of patches x number of original images
        bag_label = label[0] # refers to original image
        instance_labels = label[1] #refers to patches within original image
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, __ = model.calculate_objective(data, bag_label)
        train_loss += loss.data.item()
        error, __ = model.calculate_classification_error(data, bag_label)
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()

        #if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
            #bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
            #instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                 #np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

            #print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                  #'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    #if isinstance(train_loss, float): return np.array(train_loss) # <-- Added this line

    #if isinstance(train_loss, (np.ndarray, np.generic)): return v
    #if isinstance(v, (list,tuple)): return [to_np(o) for o in v]
    #if isinstance(v, Variable): v=v.data
    #if torch.cuda.is_available() and is_half_tensor(v): v=v.float()
    #if isinstance(v, torch.FloatTensor): v=v.float()

    train_loss = to_np(train_loss)
    #print(train_loss)

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))


def test():
    model.eval()
    test_loss = 0.
    test_error = 0.
    for batch_idx, (data, label) in enumerate(test_loader): # test loader should be of length: number of patches x number of original images
        bag_label = label[0] # refers to original image
        instance_labels = label[1] # refers to patches of original image
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights = model.calculate_objective(data, bag_label)
        test_loss += loss.data.item()
        error, predicted_label = model.calculate_classification_error(data, bag_label)
        test_error += error

        #if batch_idx < 10:  # plot bag labels and instance labels for first 5 bags
            #bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
            #instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                 #np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

            #print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                  #'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    #to_np(test_loss)

    #if isinstance(test_loss, float): return np.array(test_loss) # <-- Added this line

    test_loss = to_np(test_loss)
    #print(test_loss)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))


if __name__ == "__main__":
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    print('Start Testing')
    test()
