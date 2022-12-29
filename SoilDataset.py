import torch
import os
import cv2
from glob import glob

class SoilDataset(torch.utils.data.Dataset):

    """Soil-Residue Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir,
            masks_dir,
            transform=None, 
            preprocessing=None
    ):
        
        self.image_paths = sorted(glob(os.path.join(images_dir, '*.JPG')))
        self.mask_paths = sorted(glob(os.path.join(masks_dir, '*.PNG')))

        self.transform = transform
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        # image = cv2.imread(self.image_paths[i])
        mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
        # Convert mask to 0,1
        mask = mask/255

        data = image, mask

        # apply transform
        if self.transform:
            sample = self.transform(data)
            image, mask = sample
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image, mask = torch.from_numpy(image.copy()), torch.from_numpy(mask.copy()).unsqueeze(0)
        # return image, mask # Mask should be 1 x 256 x 256
        return image,mask
        
    def __len__(self):
        return len(self.image_paths)