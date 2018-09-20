import os

import numpy as np
from skimage import io, transform
from tqdm import tqdm

TARGET_SIZE = (499, 499, 3)
# * Training set generation for 3 datasets *

## 1. Food-5K
f_root = '/Volumes/JS/Food-5K/training'
f_train = [os.path.join(f_root, f) for f in os.listdir(f_root) if f.endswith('.jpg')]

# Read the images
X_train_3, y_train_3 = [], []

# Read the .txt file to get the image path
food101_txt_path = '/Volumes/JS/food-101/meta'
food101_img_path = '/Volumes/JS/food-101/images'
with open(food101_txt_path + '/' + 'train_split.txt', 'r') as train_split:
    for i, line in enumerate(train_split):
        if i < 10000:
            print(i)
            line = line.strip('\n')
            line = line.split('/')
            img_path = food101_img_path + '/' + line[0] + '/' + line[1] + '.jpg'
            im = io.imread(img_path)
            if len(im.shape) == 3:
                im = transform.resize(im, output_shape=TARGET_SIZE)
                im = (im * 255.0).astype(np.uint8)
                X_train_3.append(im)
                y_train_3.append(int(1))

for f_im in tqdm(f_train):
    im = io.imread(f_im)
    if len(im.shape) == 3:
        im = transform.resize(im, output_shape=TARGET_SIZE)
        im = (im * 255.0).astype(np.uint8)
        X_train_3.append(im)
        y_train_3.append(int(os.path.basename(f_im)[0]))
# X_train = np.array(X_train)
# y_train = np.array(y_train)
print('Food-5K Dataset')
print('X_train_3.shape:', len(X_train_3))
print('y_train_3.shape:', len(y_train_3))

## 2. PASCAL dataset with label all negative (non-food)
pascal_2011 = '/Volumes/JS/PASCAL2012/VOCdevkit/VOC2012/training'
pascal_train = [os.path.join(f_root, f) for f in os.listdir(f_root) if f.endswith('.jpg')]

# Read the images
for pascal_im in tqdm(pascal_train):
    im = io.imread(pascal_im)
    if len(im.shape) == 3:
        im = transform.resize(im, output_shape=TARGET_SIZE)
        im = (im * 255.0).astype(np.uint8)
        X_train_3.append(im)
        y_train_3.append(int(0))

print('Image number: (Food-5K + PASCAL Dataset)')
print('X_train_3.shape:', len(X_train_3))
print('X_train_3.shape:', len(y_train_3))

## 3. Food-101 dataset with label all positive (food)
# food101_txt_path = '/Volumes/JS/food-101/meta'
# food101_img_path = '/Volumes/JS/food-101/images'

# # Read the .txt file to get the image path
# with open(food101_txt_path + '/' + 'train_split.txt', 'r') as train_split:
#     for i, line in enumerate(train_split):
#         if i < 10000:
#             print(i)
#             line = line.strip('\n')
#             line = line.split('/')
#             img_path = food101_img_path + '/' + line[0] + '/' + line[1] + '.jpg'
#             im = io.imread(img_path)
#             if len(im.shape) == 3:
#                 im = transform.resize(im, output_shape=TARGET_SIZE)
#                 im = (im * 255.0).astype(np.uint8)
#                 X_train_3.append(im)
#                 y_train_3.append(int(1))

print('Total image number: (Food-5K + PASCAL Dataset + Food-101)')
print('X_train_3.shape:', len(X_train_3))
print('X_train_3.shape:', len(y_train_3))

# Convert np array to .npy file
X_train_3 = np.array(X_train_3)
y_train_3 = np.array(y_train_3)
np.save('./data/X_train_3.npy', X_train_3)
np.save('./data/y_train_3.npy', y_train_3)
