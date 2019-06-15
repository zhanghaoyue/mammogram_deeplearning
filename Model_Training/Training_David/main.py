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

Set paths to imaging and clinical data, balance data, preprocess data

"""

#############Set path's to images, clinical dataframe, home folder##############

mg_images = '/home/d.gordon/image_athena_no_implant/'
bc_clinical_sub = pd.read_csv('/home/d.gordon/label_athena_no_implant.csv',low_memory=False)
d_path = '/home/d.gordon/'


    ##############################Balance Data#####################################

bc_clinical_subset_1 = bc_clinical_sub.loc[bc_clinical_sub['cancer_label']==1]
bc_clinical_subset_0 = bc_clinical_sub.loc[bc_clinical_sub['cancer_label']==0].sample(n=len(bc_clinical_subset_1),random_state=546)
bc_clinical_subset = bc_clinical_subset_1.append(bc_clinical_subset_0)

    ###########################Preprocess Images###################################


    ## read in image file and apply image processing

for row in bc_clinical_subset.itertuples(index=False):
    try:
        img = cv2.imread(mg_images+row.filename+'.jpg')

        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        grayscale[grayscale>254] = 0

        ret, th = cv2.threshold(grayscale, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        bbox = cv2.boundingRect(th)

        x,y,w,h = bbox

        foreground = img[y:y+h, x:x+w]

        cv2.imwrite(d_path+'preprocessed5/'+row.filename+'.png',foreground)
    except:
        continue
        

images = []
patchess = []
for filename in os.listdir(d_path+ 'preprocessed5/'):
    img = cv2.imread(d_path + 'preprocessed5/'+filename)
    patches = image.extract_patches_2d(img,max_patches=50,patch_size=(128,128))
    images.append(img)
    patchess.append(patches)
    
bc_clinical_subset.reset_index(inplace=True,drop=True)

data = {}

for i in range(0,len(bc_clinical_subset)):
    label = bc_clinical_subset.cancer_label[i]
    patient = bc_clinical_subset.filename[i]
    for patch in patchess:
        patch_level_of_images = patch #all images(580) with 50 patches per image
        for patches in patch_level_of_images:
            patch_level_of_patch = patches #one image(1) with 50 patches per image

        # load data
            img = patch_level_of_patch # 50 patches per image value.
                #+bc_clinical_subset.filename[0]

            data[patient] = {
            'imgs':img,
            'label':label
                }
    
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(90),
    #transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],
                         std=[0.5,0.5,0.5])
    ])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],
                             std=[0.5,0.5,0.5])
    ])

dataset_transform = {'train':transform_train, 'test':transform_test}
                
    
num_folds = 10
# parameters
folds = []
output = '%s/indices_%s_fold.pkl' %(d_path,num_folds)
# check if the folder exist
if not os.path.exists(output):
# create the stratified fold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=102)
    for train_index,test_index in skf.split(bc_clinical_subset.filename,bc_clinical_subset.cancer_label):
        folds.append(bc_clinical_subset.filename[test_index][:-4].tolist())
# save the fold
        with open(output, 'wb') as l:
        pickle.dump(folds,l)
    else:
# load
        with open(output, 'rb') as l:
        folds = pickle.load(l)

# gets datasets, labels, dataloaders
        for fold_index in range(0,len(folds)):
            datasets,labels,dataloaders = get_train_test_set(folds, fold_index, data, dataset_transform)

        
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

Note: In future, instead of hardcoding the inputs for the above parameters, consider using something like int(input('enter desired number')),
may be more user friendly and allow for more flexibility as you work with different datasets.

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
