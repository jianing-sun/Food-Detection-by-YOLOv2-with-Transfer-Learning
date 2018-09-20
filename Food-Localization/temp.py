import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from keras.applications import inception_v3
from keras.applications.resnet50 import preprocess_input
from keras.layers import GlobalAveragePooling2D, Dropout, Input, Conv2D, Reshape, Lambda
from keras.applications.mobilenet import MobileNet
from keras.models import Model, load_model
from keras.preprocessing import image
from skimage import transform


# Rescale array to 0-1
def rescale_arr(arr):
    min_, max_ = np.min(arr), np.max(arr)
    arr = arr - min_
    arr = arr / (max_ - min_)
    return arr


# Preprocessing functions for images
def preprocess_im(im, input_size=(256, 256, 3)):  # 299x299
    im = transform.resize(im, output_shape=input_size)
    im = im * 255.0
    im = inception_v3.preprocess_input(im)  # tf mode: scale image from -1 to 1
    return np.expand_dims(im, axis=0)


def pretrained_path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    x = np.expand_dims(x, axis=0)
    # convert RGB -> BGR, subtract mean ImageNet pixel, and return 4D tensor
    return preprocess_input(x)


def get_model():
    base_model = construct_model()
    base_model = base_model.load_weights('./all_imgs_mobile_net_valloss0_17.h5')
    # x = Model(inputs=base_model.input, outputs=base_model.layers[-4].output)
    x = base_model.layers[2].output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.33)(x)
    model = Model(inputs=base_model.input, outputs=x)
    print(model.summary())
    # get GAP layer weights
    GAP_layer_weights = base_model.layers[-1].get_weights()[0]
    model_maps = Model(inputs=base_model.input, outputs=(base_model.layers[-4].output, base_model.layers[-1].output))
    return model_maps, GAP_layer_weights


def construct_model():
    IMAGE_H, IMAGE_W = 224, 224
    CLASS = 100
    TRUE_BOX_BUFFER = 50
    BOX = 5
    GRID_H, GRID_W = 7, 7

    input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
    true_boxes = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER, 4))

    mobilenet = MobileNet(input_shape=(224, 224, 3), include_top=False)
    x = mobilenet(input_image)
    x = Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
    output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

    # small hack to allow true_boxes to be registered when Keras build the model
    # for more information: https://github.com/fchollet/keras/issues/2790
    output = Lambda(lambda args: args[0])([output, true_boxes])

    model = Model([input_image, true_boxes], output)
    model.summary()

    layer = model.layers[-4]  # the last convolutional layer# the la
    weights = layer.get_weights()

    new_kernel = np.random.normal(size=weights[0].shape) / (GRID_H * GRID_W)
    new_bias = np.random.normal(size=weights[1].shape) / (GRID_H * GRID_W)

    layer.set_weights([new_kernel, new_bias])

    return model


# def get_model():
#     base_model = construct_model()
#     base_model = base_model.load_weights('./all_imgs_mobile_net_valloss0_17.h5')
#     # print(base_model.summary())
#     # get GAP layer weights
#     GAP_layer_weights = base_model.layers[-1].get_weights()[0]
#     model_maps = Model(inputs=base_model.input, outputs=(base_model.layers[-4].output, base_model.layers[-1].output))
#     return model_maps, GAP_layer_weights


def get_CAM(im, img_path, model, GAP_layer_weights):
    last_conv_output, pred_vec = model.predict(pretrained_path_to_tensor(img_path))
    pred = np.argmax(pred_vec)
    last_conv_output = np.squeeze(last_conv_output)
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=1)
    GAP_layer_weights = GAP_layer_weights[:, pred]  # dim: (2048,)  pred=0 -> food

    # get class activation map for object class that is predicted to be in the image
    scaled_maps = GAP_layer_weights * last_conv_output
    # final_output = np.dot(mat_for_mult.reshape((224 * 224, 2048)), GAP_layer_weights).reshape(224, 224)
    # Take mean from (5, 5, 2048) -> (5, 5)
    cam_map = np.mean(scaled_maps, axis=2)

    # Resize it back to original image size
    cam_map = rescale_arr(cam_map)
    cam_map = transform.resize(cam_map, output_shape=im.shape[:-1])

    # return class activation map
    return cam_map, pred


def plt_CAM(img_path, ax, model_maps, GAP_layer_weights):
    im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (224, 224))
    ax.imshow(im, alpha=0.5)
    CAM, pred = get_CAM(im, img_path, model_maps, GAP_layer_weights)
    ax.imshow(CAM, cmap='jet', alpha=0.5)


if __name__ == '__main__':
    model_maps, GAP_layer_weights = get_model()
    img_path = './input_img/1_1034.jpg'
    fig, ax = plt.subplots()
    plt_CAM(img_path, ax, model_maps, GAP_layer_weights)
    plt.show()
