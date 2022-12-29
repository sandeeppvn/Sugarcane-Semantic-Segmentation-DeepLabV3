import os
from glob import glob

exp_name = 'ResidueSegmentation'
for path in sorted(glob(os.path.join('./saved_models', exp_name, '*.pth'))):
    # Get the last number of the path
    epoch = path.split('_')[-1].split('.pth')[0]
    # Convert epoch to a 5-digit number
    new_epoch = '0'*(5-len(epoch))+epoch
    # Rename the file
    os.rename(path, os.path.join('./saved_models', exp_name, 'model_'+new_epoch+'.pth'))