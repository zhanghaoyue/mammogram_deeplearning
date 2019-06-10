# be223c - Mammogram Classification


# Introduction

Please put your code into the structure

* Remember to document your code in Docstring manner

* Remember to update your code running instruction here in your section


# Data Preprocessing - Yannan

1. Data Cleaning
	## Functions and methods
		### load_data: to load Athena data for this project
		### crop_image: to crop the background of an image
		### image_preprocessing: to preprocess images
		### load_implant_list: to load implant list from csv
		### create_train_data: to generate data for training
	## Description of the algorithms
		### A list of names of images with implants have been pre-identified and will be used to exclud images with implants. When matching the images with the labels, images with implants will be removed from the image list.
		### After removing images with implants, the image list needs to be further processed by removing low quality images, which have also been pre-identified and saved to a list. 
	## Expected output
		### The expected output of this algorithm is a list of preprocessed images (no images with implants and no low quality images) that are ready for using as training data for this project. 
		### All functions have been saved to a library called utils_athena. 
	## Running instructions
		### First, open run_athena.py saved in be223c/Model_Training/Data_Preprocess/. 
		### Second, import utils_athena and glob.
		### Third, set paths for importing files and saving results.
		### Fourth, run load data, load image list, and create train data blocks.
		### Finally, remove the low quality images in the low_quality_list.

2. Pectoral Muscle Removal
	# Functions and methods

	# Description of the algorithms

	# Expected output

	# Running instructions



# Model Training - Zichen



# Model Training - David


# Model Deployment - Harry

