"""
Outputs the probability of the image being a food image and
location of the food in the input image
"""
from __future__ import print_function

import cv2
import numpy as np
from keras.applications import inception_v3
from keras.models import Model, load_model
from skimage import io, transform

# INPUT_FILENAME, OUTPUT_FILENAME = sys.argv[1], sys.argv[2]
INPUT_FILENAME = './input_img/attached.jpg'
OUTPUT_FILENAME = './output_img/attached_out.jpg'


# Preprocessing functions for images
def preprocess_im(im, input_size=(299, 299, 3)):
    im = transform.resize(im, output_shape=input_size)
    im = im * 255.0
    im = inception_v3.preprocess_input(im)
    return np.expand_dims(im, axis=0)


# Rescale array to 0-1
def rescale_arr(arr):
    min_, max_ = np.min(arr), np.max(arr)
    arr = arr - min_
    arr = arr / (max_ - min_)
    return arr


# Load the model
model = load_model('./model_food.h5')
# Get the model to get the maps
model_maps = Model(inputs=model.input, outputs=model.layers[-3].input)

# Load the input image
im = io.imread(INPUT_FILENAME)

# Get the maps
maps = model_maps.predict(preprocess_im(im))
maps = np.squeeze(maps)

# Get the weights (to scale each map)
layer_weights = model.layers[-1].get_weights()[0]  # Weights of the FC layer
scaling_weights = layer_weights[:, 1]

# Multiply
scaled_maps = maps * scaling_weights

# Take mean
cam_map = np.mean(scaled_maps, axis=2)

# Resize it back to original image size
cam_map = rescale_arr(cam_map)
cam_map = transform.resize(cam_map, output_shape=im.shape[:-1])

# Show it superimposed on heatmap
heatmap = cv2.applyColorMap((cam_map * 255.0).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
im_result = (im / 255.0) * 0.5 + (heatmap / 255.0) * 0.5

# Prediction score
p = model.predict(preprocess_im(im))[0, 1]
print('Food Probability Score:', p)

# Save the image
io.imsave(OUTPUT_FILENAME, im_result)
