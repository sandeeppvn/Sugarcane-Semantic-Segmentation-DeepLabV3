import pandas as pd
import numpy as np

file_path = 'ResidueSegementation/tensorboard_logs/ResidueSegmentation/valid_logs.pkl'

# Read the pkl file
log = np.load(file_path, allow_pickle=True)

iou_scores = []
for i in range(len(log)):
    iou_scores.append(log[i]['iou_score'])

# Print max iou score
print('Max iou score: ', max(iou_scores))