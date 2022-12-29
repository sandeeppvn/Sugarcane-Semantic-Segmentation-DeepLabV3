import torch
from torch.utils.data import DataLoader
import argparse
from glob import glob
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
from SoilDataset import SoilDataset
from ResidueSegmentation_utils import *
import pickle

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='Indicate if you want to resume where you left off')
parser.add_argument('--epochs', action='store', type=int, default=5500, help='Total Number of epochs to train')
parser.add_argument('--exp_name', action='store', type=str, default='ResidueSegmentation', help='Name of experiment')
parser.add_argument('--model', action='store', type=str, default='deeplabv3plus', help='Name of model')
parser.add_argument('--backbone', action='store', type=str, default='resnet152', help='Name of model')
parser.add_argument('--activation', action='store', type=str, default='sigmoid', help='Name of model')
opt = parser.parse_args()

# exp_name = 'ResidueSegmentation'
exp_name = opt.exp_name
writer_path = os.path.join('./tensorboard_logs', exp_name)
writer = SummaryWriter(writer_path)
if not os.path.exists(os.path.join('./saved_models', exp_name)):
    os.mkdir(os.path.join('./saved_models', exp_name))

image_path = './data/Images'
mask_path = './data/Masks'

ENCODER = opt.backbone
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = opt.activation
EPOCHS = opt.epochs

# Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
TRAINING = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create segmentation model with pretrained encoder
if opt.model == 'deeplabv3plus':
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        activation=ACTIVATION,
    )
elif opt.model == 'unet':
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        activation=ACTIVATION,
    )

model.to(DEVICE)
loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5)]
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

START_EPOCH = 0
best_iou_score = 0.0
if opt.resume:
    saved_models = glob(os.path.join('./saved_models', exp_name, '*.pth'))
    epochs = [int(model_name.split('/')[-1].split('_')[-1].split('.pth')[0]) for model_name in saved_models if 'best' not in model_name]
    epochs = sorted(epochs)
    latest_epoch = epochs[-1]
    # print(latest_epoch)
    saved_model_filename = os.path.join('./saved_models', exp_name, 'model_'+str(latest_epoch)+'.pth')
    saved_model = torch.load(saved_model_filename)
    model.load_state_dict(saved_model['model_state_dict'])
    optimizer.load_state_dict(saved_model['optimizer'])
    START_EPOCH = saved_model['epoch']+1
    best_iou_score = saved_model['best_iou_score']

train_dataset = SoilDataset(
    image_path+'/Training', 
    mask_path+'/Training',
    transform=get_training_transforms(),
    preprocessing=get_prepreprocessor(smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)),
)

valid_dataset = SoilDataset(
    image_path+'/Validation', 
    mask_path+'/Validation',
    preprocessing=get_prepreprocessor(smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)),
)


# Get train and val data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=5, 
    shuffle=True, 
    num_workers=0
)
valid_loader = DataLoader(
    valid_dataset, 
    batch_size=2, 
    shuffle=False, 
    num_workers=0
)

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)
valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)


if TRAINING:
    
    train_logs_list, valid_logs_list = [], []
    best_loss = float('inf')

    for i in range(START_EPOCH, EPOCHS):
        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        train_logs_list.append(train_logs)

        
        if i%100 == 99:
            valid_logs = valid_epoch.run(valid_loader)
            valid_logs_list.append(valid_logs)
            writer.add_scalar('valid/IoU', valid_logs['iou_score'], i)
            writer.add_scalar('valid/loss', valid_logs['dice_loss'], i)
            save_dict = {
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_iou_score': best_iou_score,
                'encoder': ENCODER,
                'activation': ACTIVATION,
            }
            torch.save(save_dict, os.path.join('./saved_models', exp_name, 'model_'+str(i)+'.pth'))

            # Save model if a better val IoU score is obtained
            if best_iou_score < valid_logs['iou_score']:
                best_iou_score = valid_logs['iou_score']
                best_loss = valid_logs['dice_loss']
                os.system('cp ' + os.path.join('./saved_models', exp_name,'model_'+str(i)+'.pth') + ' ' + os.path.join('./saved_models', exp_name, 'model_best.pth'))
                print('New Best Model found and saved!')

        # Update learning rate if epoch is divisible by 10
        if i % 750 == 0:
            optimizer.param_groups[0]['lr'] /= 2

        # Add logs to TensorBoard
        writer.add_scalar('train/IoU', train_logs['iou_score'], i)
        writer.add_scalar('train/loss', train_logs['dice_loss'], i)


print('\nBest_iou_score: {}'.format(best_iou_score))
print('Best_loss: {}'.format(best_loss))

with open(os.path.join(writer_path, 'train_logs.pkl'), 'wb') as f:
    pickle.dump(train_logs_list, f)

with open(os.path.join(writer_path, 'valid_logs.pkl'), 'wb') as f:
    pickle.dump(valid_logs_list, f)

writer.close()
print('Training Complete!!')