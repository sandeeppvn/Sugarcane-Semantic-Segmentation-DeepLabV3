from ResidueSegmentation_utils import *
import math, random, os, cv2
import numpy as np
from tqdm import tqdm

image_path = './data/Images'
mask_path = './data/Masks'

recreate_dirs(image_path)
recreate_dirs(mask_path)

old_image_path = './data/Original_Images'
old_mask_path = './data/Original_Masks'

homogenoues,heterogenous = 0,0
for filename in tqdm(os.listdir(old_image_path)):
    if filename.endswith('.JPG'):
        image = cv2.imread(os.path.join(old_image_path, filename)) 
        image = cv2.rotate(image, cv2.ROTATE_180) 

        mask_filename = filename[:-4]+'.PNG'
        # Read mask as grayscale
        mask = cv2.imread(os.path.join(old_mask_path, mask_filename), cv2.IMREAD_GRAYSCALE)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        
        # The cropped image must be divisible by 32
        mask = center_crop(mask, (3552,2688))
        image = center_crop(image, (3552,2688))

        if check_homogenity(mask):
            heterogenous+=1
            cv2.imwrite(os.path.join(mask_path+'/Training', mask_filename), mask)
            cv2.imwrite(os.path.join(image_path+'/Training', filename), image)
        else:
            homogenoues+=1

print('Total Non homogenous images:',heterogenous)
print('Total Homogenous images:',homogenoues)
print()

path_names = []
for filename in os.listdir(image_path+'/Training'):
    if '.JPG' in filename:
        image_pathname = image_path+'/Training/'+filename
        mask_pathname = mask_path +'/Training/'+filename[:-4]+'.PNG'
        path_names.append((image_pathname,mask_pathname))


valid_size = math.ceil(len(path_names)*0.1)
test_size = math.ceil(len(path_names)*0.1)

print('Training Dataset size:',len(path_names)-valid_size-test_size)
print('Validation Dataset size:',valid_size)
print('Testing Dataset size:',test_size)

for i in range(valid_size):
    random_idx = random.randint(0, len(path_names)-1)
    image,mask = path_names[random_idx]
    path_names.remove(path_names[random_idx])
    os.rename(image, image.replace('Training', 'Validation'))
    os.rename(mask, mask.replace('Training', 'Validation'))
for i in range(test_size):
    random_idx = random.randint(0, len(path_names)-1)
    image,mask = path_names[random_idx]
    path_names.remove(path_names[random_idx])
    os.rename(image, image.replace('Training', 'Testing'))
    os.rename(mask, mask.replace('Training', 'Testing'))

print('Data Preparation Completed!')