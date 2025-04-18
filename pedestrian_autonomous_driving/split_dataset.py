"""
Splits a dataset of images into training and testing subsets.

This script organizes a dataset into a train/test split by copying image files
from a source directory into two target subdirectories (train and test), randomly shuffling the images and splits them
according to 80-20 ratio.

Dependencies
------------
- os : For directory and path operations.
- shutil : For copying files.
- random : For shuffling the image list to ensure a random split.

Parameters
----------
source_dir : str
    Path to the original dataset directory.
target_dir : str
    Path where train/test split will be stored.
train_ratio : float
    Ratio of the dataset to use for training (0.8).

Returns
-------
None
    Copies files into `train` and `test` subdirectories.
"""
import os
import shutil
import random

source_dir = 'E:/facultate/licenta/pda_backup/PedestrianAutonomousDriving/pedestrian_autonomous_driving/dataset'
target_dir = 'E:/facultate/licenta/pda_backup/PedestrianAutonomousDriving/pedestrian_autonomous_driving/weather_dataset'
train_ratio = 0.8

train_dir = os.path.join(target_dir, 'train')
test_dir = os.path.join(target_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]

    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_dir, class_name, img)
        shutil.copy2(src, dst)

    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(test_dir, class_name, img)
        shutil.copy2(src, dst)

print("Dataset split into 80% train and 20% test.")
