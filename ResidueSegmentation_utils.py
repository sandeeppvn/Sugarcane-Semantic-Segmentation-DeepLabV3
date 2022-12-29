################################################################################
#           List of Utils functions for ResidueSegmentation_utils.py           #
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import torchvision.transforms as transforms
import albumentations as album
from Transform import *

import warnings
warnings.filterwarnings("ignore")


#HELPER Functions
################################################################################
#                               Data Preparation                               #
################################################################################
def recreate_dirs(path):
    if os.path.exists(path+'/Training'):
        shutil.rmtree(path+'/Training', ignore_errors=True)
    os.makedirs(path+'/Training')

    if os.path.exists(path+'/Validation'):
        shutil.rmtree(path+'/Validation', ignore_errors=True)
    os.makedirs(path+'/Validation')

    if os.path.exists(path+'/Testing'):
        shutil.rmtree(path+'/Testing', ignore_errors=True)
    os.makedirs(path+'/Testing')

################################################################################
#                                Image Preprocessing                           #
################################################################################
def center_crop(img, dim):
    """Returns center cropped image
    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped
    """
    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img

def check_homogenity(mask):
    """Returns True if image is homogeneous
    Args:
    mask: one hot encoded mask to be checked
    """
    mask_size = 1
    for i in mask.shape:
        mask_size *=i    
    val = (np.count_nonzero(mask) / mask_size)
    # print(val)
    return 0.1 < val < 0.9


################################################################################                                                                            
#                                Visualization                                 #
################################################################################
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

################################################################################
#                                Data Augmentation                             #
################################################################################

def get_training_transforms():
    """
    Data augmentation configuration for training
    
    # Returns
        An imgaug object that can be used to augment images
    """
    train_transform = [    
        CustomRandomCrop(size=(256,256)),
        CustomRandomHorizontalFlip(),
        CustomRandomVerticalFlip(),
        CustomRandomRotation90(),
    ]
    return transforms.Compose(train_transform)

def get_validation_transforms():   
    """
    Data augmentation configuration for validation
    Add sufficient padding to ensure image is divisible by 32

    # Returns
        A valid imgaug object that can be used to augment images
    """
    return transforms.Compose ([
        CustomRandomCrop(size=(256,256)),
    ])

################################################################################
#                                Data Preprocessing                            #
################################################################################

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_prepreprocessor(preprocessing_fn=None):
    trans = []
    if preprocessing_fn:
        trans.append(
            album.Lambda(image=preprocessing_fn)
        )
    trans.append(album.Lambda(image=to_tensor))
    return album.Compose(trans)
