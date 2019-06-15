#############################Import Packages###################################

import torch.utils.data as data
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import collections
import itertools
import get_preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
from sklearn.model_selection import StratifiedKFold
import pickle
import re
import io
import csv
import cv2
import random
import glob
import imageio

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

import SimpleITK as sitk

##########################Pytorch Dataset Class#################################

class MGDataset(data.Dataset):
    """
    Pytorch Dataset class.
    """

    def __init__(self, data, patients, transform=None, datasetType='Train'):
        """
        :data: data dictionary containing patches for images and image/bag level label for patients. In the form: data['imgs'], data['label']
        :patients: patients to be used
        :transform: the transform for the data
        :datasetType: the type of the data, train or test
        """
        self.data = data
        self.patients = patients
        self.transform = transform
        self.datasetType = datasetType

        self.total_patients = len(self.patients)


    def __getitem__(self, index):
        """
        getitem for pytorch dataloader
        """
        if self.datasetType == 'Train':
            actual_index = index % self.total_patients

            idx = self.patients[actual_index]
            label = self.data[idx]['label']
        else:
            idx = self.patients[index]
            label = self.data[idx]['label']

        # images
        img = self.data[idx]['imgs']

        # transform data
        if self.transform:
            img = self.transform(img)


        # sample
        sample = {'idx':idx,'img': img, 'label': label}

        #return sample
        return idx, img, label

    def __len__(self):
        return self.total_patients


##############################Show Patches#####################################

"""
Function to show a list of images
"""

def show_images(images, cols = 4, titles = None):
   """
   Display a list of images in a single figure with matplotlib.

   Parameters:
   
   :images: List of np.arrays compatible with plt.imshow.

   :cols (Default = 1): Number of columns in figure (number of rows is
                       set to np.ceil(n_images/float(cols))).

   :titles: List of titles corresponding to each image. Must have
           the same length as titles.
   """
   assert((titles is None)or (len(images) == len(titles)))
   n_images = len(images)
   if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
   fig = plt.figure()
   for n, (image, title) in enumerate(zip(images, titles)):
       a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
       if image.ndim == 2:
           plt.gray()
       plt.imshow(image, cmap='gray')
       a.set_title(title, fontsize = 100)
   fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
   plt.show()

##########################Train Test Split######################################


def get_train_test_set(folds, fold_index, data, dataset_transform):
        testset_patients = folds[fold_index]
        trainset_patients = [x for i,x in enumerate(folds) if i!=fold_index]
        trainset_patients = list(itertools.chain.from_iterable(trainset_patients))


        # Train set
        trainset = get_dataset.MGDataset(data, trainset_patients,
                                      dataset_transform['train'],
                                      datasetType='Train'
                                      )
        trainset_label = np.array([data[idx]['label'] for idx in trainset_patients])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=522, #batch size in this dataloader is = num of training bags... dataloader in the mg_bag_loader file has batch_size = 1, which propagates through the network...
                                              shuffle=True, num_workers=4)

        # Test set
        testset = get_dataset.MGDataset(data, testset_patients,
                                     dataset_transform['test'],
                                     datasetType='Test'
                                     )
        testset_label = np.array([data[idx]['label'] for idx in testset_patients])
        testloader = torch.utils.data.DataLoader(testset, batch_size=58, #batch size in this dataloader is = num of test bags... dataloader in the mg_bag_loader file has batch_size = 1, which propagates through the network...
                                              shuffle=False, num_workers=4)

        datasets = {'train':trainset, 'test':testset}
        labels = {'train':trainset_label, 'test':testset_label}
        dataloaders = {'train':trainloader,'test':testloader}

        return datasets, labels, dataloaders
