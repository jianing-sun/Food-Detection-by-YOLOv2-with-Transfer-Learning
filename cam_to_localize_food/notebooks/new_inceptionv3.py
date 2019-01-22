import numpy as np
from skimage import io
from keras.models import Model, load_model
from keras.applications import inception_v3
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import losses, optimizers
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from sklearn import metrics
import matplotlib.pyplot as plt

RANDOM_SEED = 43
np.random.seed(RANDOM_SEED)


# ** Make model **
# Base model
# base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(448, 448, 3))
base_model = inception_v3.myInceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
print(base_model.summary())
model_maps = Model(inputs=base_model.input, outputs=base_model.layers[-84].input)
print(model_maps.summary())
# Model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.33)(x)
x = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)

print(model.summary())
