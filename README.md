# Residue Segmentation

### This project is designed to perform semantic segmentation and classify Residue from Soil for a given set of sugarcane crop images and masks. DeepLabV3plus model is utilized to perform the segmentation.

## Results
### Original Image - Manual Mask - Predicted Mask
![result1](outputs/ResidueSegmentation/IMG_9689.PNG)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install XXX.

```bash
pip install XXX
```

## Usage
### PART 1: Data Setup
- The data required for training is obtained using Manually annotated images on LabelBox. LabelBox provides a python API to download images and corresponding masks. Run the following to obtain the images and masks. 
```bash
python downloadImages.py
```
*This ensures that the data is downloaded and placed in the folders:
__data/Original_Images__ and __data/Original_Masks__*

- The data needs to be processed into Training - Validation - Test split (Ratio: 80-10-10). Additionally, only the images which are not homogenous will be used in order to enable the model to learn differentiable features. To perform this, run the following.
```bash
python ResidueSegmentation_DataPreparation.py
```
*After this, non Homogenous training, validation and testing images will be present at 
__data/Images/Training__, __data/Masks/Training__,__data/Images/Validation__, __data/Masks/Validation__,__data/Images/Testing__, __data/Masks/Testing__*

### PART 2: Training
Since the training and validation data is ready, perform training of the model using the following code:
```bash
python ResidueSegmentation_Training.py --epochs 5500 --exp_name ResidueSegmentation --model deeplabv3plus --backbone resnet152 --activation sigmoid
```
*Feel free to play around with the input parameters. Eg: model: unet, epochs 11000*

This should take a couple of hours depending on the processing capacity of the computer and model used.

Upon completion of training, epoch/100 (default: 55) models + 1 best model (model_best.pth) will be saved in __saved_model/exp_name/*__

#### Tensorboard Logs
During training, the IOU score and loss metrics can be visualized in Tensorboard. To view them run the following: (replace exp_name with the appropriate value)
```bash
tensorboard --logdir tensorboard_logs/exp_name
```
The logs would be visible at http://localhost:6006/ on a browser.

### PART 3: Testing/Predictions
- To perform testing of the test dataset using only the best model, run:
```bash
python ResidueSegmentation_Testing.py
```
This will generate the predicted masks for the test images and store them at __outputs/exp_name/*__

- To Analyse how the model evolves over multiple epochs, GIF images consisting of sequential predicted images are generated for the test data. The logs can also be visualized on tensorboard. To generate this, run:
```bash
python ResidueSegmentation_Predictions.py
tensorboard --logdir tensorboard_logs/exp_name
```
The logs would be visible at http://localhost:6006/ on a browser.
The generated GIF image will be present at __outputs/GIF/exp_name__ for each test image.


## Author
Sandeep Polavarapu Venkata Naga

## Contributors
Varaprasad Bandaru
Koutilya PNVR
Rajeshwar Natarajanshanmugasingaram


## License
