# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:57:35 2019

@author: YannanLin
"""

import pandas as pd
from PIL import Image
import numpy as np
import os
from skimage.filters import threshold_triangle
from skimage.morphology import erosion, square
import matplotlib

def load_data(file_path):
    """
    Function to load data for this project
    
    Input: file_path
    Output:file_df, new_df, filename, cancer, name_cancer_dic, name_view_dic
    
    This function takes the path to all the images as input
    and outputs the following:
        
    file_df is the dataframe of the provided csv file
    new_df is the datafram including only MLO and CC view images
    cancer is a vector of labels for images
    name_cancer_dic is a dictionary of which image name is the key and label is the value
    name_view_dic is a disctionary of which image name is the key and view is the value
    
    """    
    # load csv file
    file = pd.read_csv(file_path)
    file_df = pd.DataFrame(file)

    # subset only CC and MLO views
    new_df = file_df[(file_df['view']=='CC') | (file_df['view']=="MLO")]
    #5147 rows x 12 columns

    filename = new_df["filename"].tolist()
    cancer_left = new_df["Cancer_L"].tolist()
    cancer_right = new_df["Cancer_R"].tolist()
    view = new_df["view"].tolist()

    cancer = [0]*len(filename)
    for i in range(len(cancer_left)):
        if cancer_left[i] == 0 and cancer_right[i] == 0:
            continue
        else: 
            cancer[i] = 1

    name_cancer_dic = {}

    for i in range(len(filename)):
        name_cancer_dic[filename[i]] = cancer[i]
         
    name_view_dic = {}
    
    for i in range(len(filename)):
        name_view_dic[filename[i]] = view[i]
        
    return file_df, new_df, filename, cancer, name_cancer_dic, name_view_dic

def crop_image(image,tol=0):
    """
    Function to crop the background of an image
    
    Input: img, tol
    Output: cropped_image
    
    img is the input image
    tol is tolorance with default value 0
    cropped_image is the image without redundant background
    
    """
    mask = image>tol   
    return image[np.ix_(mask.any(1),mask.any(0))]

def image_preprocessing(image):
    """
    Function to preprocess images
    
    Input: image
    Output: image
    
    input image is a raw input image
    output image is the preprocessed image
    
    Steps of preprocessing:
    1. Standardization
    for all pixels
    first subtract mean of the image
    then divided by standard deviation of the image
    
    2. Thresholding
    apply triangle method for thresholding
    generate a binary mask
    region of breast has value of 1
    region of background has value of 0    
    
    3. Fill holes
    apply erosion to fill small holes on the mask
    
    4. Apply mask to image
    multiply mask to image such that
    background region has value of 0
    breast regoin has non-zero values
    
    5. Crop out background
    use crop_image() function to crop out background
    
    """
    mean = np.mean(image)
    std = np.std(image)
    image = image-mean
    image = image/std
    
    image_copy = image

    thresh = threshold_triangle(image)
    binary = np.where(image > thresh,1.0,0.0)
    
    filled = erosion(binary, square(15))
    
    img = filled*image_copy
    
    image = crop_image(img,0)
    
    return image

def load_implant_list(implant_list_path):
    """
    Function to load implant list from csv
    
    Input: implant_list_path
    output: implant_list
    
    This function takes the path to the csv file containing
    all names of images with implants as input and outputs:
        
    implant_list contains the names of the images that have
    implants, which have been pre-identified and stored in 
    the implant.csv file.
    
    """
    file = pd.read_csv(implant_list_path)
    file_df = pd.DataFrame(file)
    
    implant_list = file_df["implant"].tolist()
    
    for i in range(len(implant_list)):
        implant_list[i] = implant_list[i].replace('"', '')
    
    return implant_list

def create_train_data(image_path_list, implant_list, filename, name_cancer_dic,
                      image_implant_raw_path,image_athena_no_implant_path,
                      name_view_dic, MLO_path):
    
    """
    Function to generate data for training
    
    Input: image_path_list, implant_list, filename, name_cancer_dic,
           image_implant_raw_path,image_athena_no_implant_path,
           name_view_dic, MLO_path
    Output: image_list, label_list, name_list, 
            label_list_implant, image_list_implant
    
    This function selects images from the image folder that has
    a corresponding label in the csv file and seperates images
    with and without implants using a list of image names 
    (implant_list). It also save MLO view images into a folder.
    
    image_path_list is a list of paths to images
    implant_list is a list of names of images with implants
    filename is a list of file names of all images provided
    name_cancer_dic is a dictionary of which the key is the name 
    of the image and the value is 0 (no cancer) or 1 (cancer)
    name_view_dic is a dictionary of which the key is the name 
    of the image and the value is the MLO or CC
    image_implant_raw_path is the path to the images with implants
    image_athena_no_implant_path is the path to the images without
    impoants
    MLO_path is the path to the MLO view images
    
    
    image_list is a list of images without implants.
    label_list is a list of labels correspnonding to image_list.
    name_list is a list of image names without implants. 
    image_list_implant is a list of images with implants.
    label_list_implant is a list of labels corresponding to the 
    image_list_implant.
    
    """
    
    label_list = []
    image_list = []
    name_list = []
    
    label_list_implant = []
    image_list_implant = []
 
    temp_filename = filename #5147

    for i in range(len(image_path_list)):
        print(i)
        name = image_path_list[i].split("/")[-1]
        image_name = name.split(".")[0]

        if image_name in temp_filename:
            if image_name in implant_list:
                img = Image.open(image_path_list[i])
                img = image_preprocessing(img)

                label_list.append(name_cancer_dic[image_name])
                image_list_implant.append(img)

                temp_filename.remove(image_name)

                matplotlib.image.imsave(os.path.join(image_implant_raw_path+image_name+'.jpg'), img)
                continue

            else:
                img = Image.open(image_path_list[i])
                img = np.asarray(img)
                img = image_preprocessing(img)

                label_list.append(name_cancer_dic[image_name])
                image_list.append(img)
                name_list.append(image_name)

                temp_filename.remove(image_name)

                matplotlib.image.imsave(os.path.join(image_athena_no_implant_path+image_name+'.jpg'), img)
                
                if name_view_dic[image_name] == "MLO":
                    matplotlib.image.imsave(os.path.join(MLO_path+image_name+'.jpg'), img)

    return image_list, label_list, name_list, label_list_implant, image_list_implant