# be223c - Mammogram Classification


# Introduction

Please put your code into the structure

* Remember to document your code in Docstring manner

* Remember to update your code running instruction here in your section


# Data Preprocessing - Yannan

1. Data Cleaning
	- Functions and methods
		- load_data: to load Athena data for this project
		- crop_image: to crop the background of an image
		- image_preprocessing: to preprocess images
		- load_implant_list: to load implant list from csv
		- create_train_data: to generate data for training
	- Description of the algorithms
		- A list of names of images with implants have been pre-identified and will be used to exclude images with implants. When matching the images with the labels, images with implants will be removed from the image list.
		- After removing images with implants, the image list needs to be further processed by removing low quality images, which have also been pre-identified and saved to a list. 
		- All functions have been saved to a library called utils_athena. 
	- Expected output
		- The expected output of this algorithm is a list of preprocessed images (no images with implants and no low quality images) that are ready for using as training data for this project. 
	- Running instructions
		- First, open run_athena.py saved in be223c/Model_Training/Data_Preprocess/. 
		- Second, import utils_athena and glob.
		- Third, set paths for importing files and saving results.
		- Fourth, run load data, load image list, and create train data blocks subsequently.
		- Finally, remove the low quality images in the low_quality_list.

2. Pectoral Muscle Removal
	- Functions and methods
		- create_train_data: to create traning data
		- rotate: to rotate an image
		- rotate_images: to rotate two lists of images
		- flip: to flip two lists of images
		- unet: to build u-net model
		- plot_acc_los: to plot loss and accuracy during training and validation
 		- read_test_images: to read in testing images
		- test_model: to test the model
		- print_test_result: to print out an example of test result
		- read_test_results: to read in test results	
		- mean_iou: to calculate Jaccard index
		- post_processing: to post-process result images
		- plot_post_process_results: to plot the results after post-processing
	- Description of the algorithms
		- This approach trains a u-net model to help segment pectoral muscles on digital mammograms.
		- It has four components: training, testing, evaluation, and post-processing. A classification model is trained and saved to the specified path. Fifty images are used as testing images and the predicted masks are generated for them. Jaccard index is used to evluate the performance of the model by comparing the similarity of the predicted mask and the groud truth (pre-defined). Highly unstatisfactory masks will undergo the post-processing procedure to improve the result. 
		- All functions have been saved to a library called utils_library. 
	- Expected output
		- The best model (model.h5) during training should be saved to the specified directory and used to generate predicted masks. 
	- Running instructions
		- First, open run_muscle.py saved in be223c/Model_Training/Data_Preprocess/. 
		- Second, import utils_muscle, glob, numpy, os, tensorflow, keras, and sklearn.
		- Third, set paths for importing files and saving results.
		- Fourth, run training, testing, evaluation, and post-processing blocks subsequently.

# Model Training - Zichen



# Model Training - David


# Model Deployment - Harry

## 1. Tools
   Backend:Flask, Pytorch
   
   Frontend: uikit, Vue.js, JQuery.js
   
   Deployment: Gunicorn, Docker

## 2. Code structure
   ```
   flask-app/
   ├── server.py   # contains major code of flask app
   ├── wsgi.py     # main run
   ├── Dockerfile  # use this to generate docker image
   ├── model.py    # pytorch model prediction functions
   ├── requirements.txt # all required libraries
   ├── config.py   # Flask.app config
   ├── code_zichen/ # all Zichen's model source code
   ├── templates/  # all html pages
       ├── front_page.html # introduction page
       ├── index.html # initial loading page
       ├── team_member.html # list of team members
       ├── upload.html # all upload and prediction, display functions
   ├── uploads/ # for preload saved models
   ├── venv/ # virtual env for development
   ├── static/ # all the javascript library and static files, css, uikit
   
   ```
   
## 3. Build from source (Recommended)

This is recommended because the base image is from Nvidia
and it is very big (7Gb), it's faster to directly build
than transfer the image.

Run the cmd from terminal in flask-app folder
    
    sudo docker build -t flask-app .
    
The Dockerfile not only contains a Ubuntu system with CUDA and pytorch but also
include the source code for the app.
Use the generated Docker image to run the flask app

## 4. Run the Flask app with Docker image

4.1 Download or pull image from Docker Hub.

go to hub.docker.com
search for: zhanghaoyue/flask-app-mammogram


4.2 Run the following code
    
    # load the docker image if not built from source
    sudo docker load -i flask-app.tar
    # run the docker image
    sudo docker run --runtime=nvidia -p 5000:5000 flask-app


## 5. How to use the webpage

There are 3 sub pages of the website. You can nagivate 
using the navbar function.

Introduction contains model introduction

Model Prediction contains the major function:

Upload: select a input Mamogram image
Predict: click this button for result

after click the result, three images will show up and a 
prediction result will be shown at the bottom.

- The first image is the original input image.

- The second image is the heatmap in grayscale

- the third image is the original image overlay with heatmap

The probability values and the label shows Cancer/No Cancer result

    