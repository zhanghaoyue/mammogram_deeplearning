import matplotlib.pyplot as plt
import skimage
import numpy as np
from skimage import io, morphology, measure, color, transform
from skimage.filters import threshold_otsu
import glob
import multiprocessing
import time

def apply_otsu_segmentation(img):
    thresh = threshold_otsu(img)
    binary = img > thresh
    return binary


def apply_morphology_operation(binary, operator_size=35):
    binary_dil = morphology.binary_dilation(
        binary, morphology.square(operator_size))
    return binary_dil


def locate_max_region(binary):
    label_image = measure.label(binary)
    region_list = measure.regionprops(label_image)
    region_area_list = np.array([region.area for region in region_list])
    region_max_index = np.argmax(region_area_list)
    region_max_box = region_list[region_max_index].bbox
    minr, minc, maxr, maxc = region_max_box
    return (minr, minc, maxr, maxc)


def apply_preprocess(img, rescale=5):
    img_resize = transform.rescale(img, 1/rescale, multichannel=False)
    img_resize_binary = apply_otsu_segmentation(img_resize)
    img_resize_morphology = apply_morphology_operation(img_resize_binary)
    minr, minc, maxr, maxc = locate_max_region(img_resize_morphology)
    img_crop = img[minr*rescale:maxr*rescale, minc*rescale:maxc*rescale]
    mask_resize_crop = img_resize_morphology[minr:maxr, minc:maxc]
    mask_crop = transform.resize(mask_resize_crop, img_crop.shape)
    img_crop_denoise = img_crop * mask_crop
    return img_crop_denoise


def save_preprocess(img_path):
    img_path_preprocess = img_path.replace('images', 'preprocess')
    img = io.imread(img_path)
    img_preprocess = apply_preprocess(img)
    io.imsave(img_path_preprocess, img_preprocess.astype('uint8'))
    return


if __name__ == "__main__":
    start = time.time()
    img_path_list = glob.glob('/data/zcwang/BE223c/data/images/*.png')
    pool = multiprocessing.Pool(processes=16)
    for img_path in img_path_list:
        pool.apply_async(save_preprocess, (img_path, ))
    pool.close() 
    pool.join()
    end = time.time()
    print("Sub-processes done. %f s used"%(end-start))