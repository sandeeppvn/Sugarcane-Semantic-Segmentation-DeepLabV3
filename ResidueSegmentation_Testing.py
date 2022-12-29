#RUN this file after ResidueSegmentation_Training.py
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from SoilDataset import SoilDataset
from ResidueSegmentation_utils import *

from tqdm import tqdm
from glob import glob

from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', action='store', type=str, default='ResidueSegmentation', help='Name of experiment')
parser.add_argument('--model' , action='store', type=str, default='DeepLabV3Plus', help='Name of model')
opt = parser.parse_args()

exp_name = opt.exp_name
writer_path = os.path.join('./tensorboard_logs', exp_name)
writer = SummaryWriter(writer_path)

image_path = './data/Images'
mask_path = './data/Masks'

# Create a dictionary of image name to index mapping
index_to_mask_name_dict = {}
for i,mask_name in enumerate(sorted(glob(os.path.join(mask_path+'/Testing', '*.PNG')))):
    index_to_mask_name_dict[i] = mask_name

# Get the best model to obtain model parameters
best_model_path = os.path.join('./saved_models', exp_name, 'model_best.pth')
saved_model = torch.load(best_model_path)
ENCODER = saved_model['encoder']
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = saved_model['activation']

# Create preprocessing function with pretrained imagenet encoder
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# create segmentation model with pretrained encoder
if opt.model == 'DeepLabV3Plus':
    best_model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        activation=ACTIVATION,
    )
elif opt.model == 'Unet':
    best_model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        activation=ACTIVATION,
    )

best_model.load_state_dict(saved_model['model_state_dict'])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = best_model.to(DEVICE)
best_model.eval()

# Obtain test dataset
test_dataset = SoilDataset(
    image_path+'/Testing',
    mask_path+'/Testing',
    preprocessing=get_prepreprocessor(preprocessing_fn)   
)
test_dataloader = DataLoader(test_dataset)

# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=smp.utils.losses.DiceLoss(),
    metrics=[smp.utils.metrics.IoU(threshold=0.5)],
    device=DEVICE,
)

results = test_epoch.run(test_dataloader)
writer.add_scalar('test/IoU', results['iou_score'])
writer.add_scalar('test/loss', results['dice_loss'])

writer.close()

# Display results
print(f"Test results: {results}")

# Generate predictions for the best model and store them
if not os.path.exists(os.path.join('./outputs', exp_name)):
    os.makedirs(os.path.join('./outputs', exp_name))
else:
    shutil.rmtree(os.path.join('./outputs', exp_name))
    os.makedirs(os.path.join('./outputs', exp_name))

with torch.no_grad():
    for idx,(image,mask) in enumerate(tqdm(test_dataloader)):
        
        image, mask = image.to(DEVICE), mask.to(DEVICE)
        
        output = best_model.predict(image)
        output = (output.squeeze().cpu().numpy().round()).astype(np.uint8)
        output = output*255
        output = cv2.copyMakeBorder(output, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[0, 255, 0])

        mk = mask.squeeze().cpu().numpy()
        mk = mk*255
        mk = cv2.copyMakeBorder(mk, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[0, 255, 0])
        mk = cv2.resize(mk, (output.shape[1], output.shape[0]))

        
        im = cv2.imread(index_to_mask_name_dict[idx].replace('Masks','Images').replace('PNG','JPG'))
        # im = np.transpose(im, (1,2,0))
        im = cv2.resize(im, (output.shape[1], output.shape[0]))
        
        # Save mask as png image
        save_path = os.path.join('./outputs', exp_name, '{}'.format(index_to_mask_name_dict[idx].split('/')[-1]))

        d_output = np.dstack((output, output, output))
        d_mk = np.dstack((mk, mk, mk))
        res = np.concatenate((im, d_mk, d_output), axis = 1)

        cv2.imwrite(save_path, res)

