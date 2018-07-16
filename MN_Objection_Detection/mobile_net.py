import warnings

import keras.backend as K
import keras.utils
from keras import layers
from keras import models


def relu6(x):
    return K.relu(x, max_value=6)


def MobileNetV1(input_shape,
                alpha=1.0,
                depth_multiplier=1,
                include_top=True,
                weights='imagenet',
                pooling=None,
                ):
    if input_shape[-1] not in [1, 3]:
        warnings.warn('Images must have 3 channels (RGB) or 1 channel')
    assert input_shape[0] in [224, 192, 160, 128]
    assert input_shape[1] in [224, 192, 160, 128]

    ''' construct mobilenet '''
    inputs = layers.Input(shape=input_shape)

    # Conv / s2: filter shape (3 x 3 x 3 x 32)
    num_filters = int(alpha * 32)
    x = layers.ZeroPadding2D(padding=(1, 1), name='conv1_zeropad')(inputs)
    x = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), padding='valid', use_bias=False, strides=(2, 2),
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation(relu6, name='conv1_relu6')(x)

    # Conv dw / s1: filter shape (3 x 3 x 32 dw)
    x = depthwise_block(x, kernel=(3, 3), stride=(1, 1), depth_multiplier=depth_multiplier, id=1)
    # Conv pw / s1: filter shape (1 x 1 x 32 x 64)
    x = pointwise_block(x, num_filters=64, kernel=(1, 1), stride=(1, 1), alpha=alpha, id=1)

    # Conv dw / s2: filter shape (3 x 3 x 64 dw)
    x = depthwise_block(x, kernel=(3, 3), stride=(2, 2), depth_multiplier=depth_multiplier, id=2)
    # Conv pw / s1: filter shape (1 x 1 x 64 x 128)
    x = pointwise_block(x, num_filters=128, kernel=(1, 1), stride=(1, 1), alpha=alpha, id=2)

    # Conv dw / s1: filter shape (3 x 3 x 128 dw)
    x = depthwise_block(x, kernel=(3, 3), stride=(1, 1), depth_multiplier=depth_multiplier, id=3)
    # Conv pw / s1: filter shape (1 x 1 x 128 x 128)
    x = pointwise_block(x, num_filters=128, kernel=(1, 1), stride=(1, 1), alpha=alpha, id=3)

    # Conv dw / s2: filter shape (3 x 3 x 128 dw)
    x = depthwise_block(x, kernel=(3, 3), stride=(2, 2), depth_multiplier=depth_multiplier, id=4)
    # Conv pw / s1: filter shape (1 x 1 x 128 x 256)
    x = pointwise_block(x, num_filters=256, kernel=(1, 1), stride=(1, 1), alpha=alpha, id=4)

    # Conv dw / s1: filter shape (3 x 3 x 256 dw)
    x = depthwise_block(x, kernel=(3, 3), stride=(1, 1), depth_multiplier=depth_multiplier, id=5)
    # Conv pw / s1: filter shape (1 x 1 x 256 x 256)
    x = pointwise_block(x, num_filters=256, kernel=(1, 1), stride=(1, 1), alpha=alpha, id=5)

    # Conv dw / s2: filter shape (3 x 3 x 256 dw)
    x = depthwise_block(x, kernel=(3, 3), stride=(2, 2), depth_multiplier=depth_multiplier, id=6)
    # Conv pw / s1: filter shape (1 x 1 x 256 x 512)
    x = pointwise_block(x, num_filters=512, kernel=(1, 1), stride=(1, 1), alpha=alpha, id=6)

    # repeat 5 times
    x = depthwise_block(x, kernel=(3, 3), stride=(1, 1), depth_multiplier=depth_multiplier, id=7)
    x = pointwise_block(x, num_filters=512, kernel=(1, 1), stride=(1, 1), alpha=alpha, id=7)
    x = depthwise_block(x, kernel=(3, 3), stride=(1, 1), depth_multiplier=depth_multiplier, id=8)
    x = pointwise_block(x, num_filters=512, kernel=(1, 1), stride=(1, 1), alpha=alpha, id=8)
    x = depthwise_block(x, kernel=(3, 3), stride=(1, 1), depth_multiplier=depth_multiplier, id=9)
    x = pointwise_block(x, num_filters=512, kernel=(1, 1), stride=(1, 1), alpha=alpha, id=9)
    x = depthwise_block(x, kernel=(3, 3), stride=(1, 1), depth_multiplier=depth_multiplier, id=10)
    x = pointwise_block(x, num_filters=512, kernel=(1, 1), stride=(1, 1), alpha=alpha, id=10)
    x = depthwise_block(x, kernel=(3, 3), stride=(1, 1), depth_multiplier=depth_multiplier, id=11)
    x = pointwise_block(x, num_filters=512, kernel=(1, 1), stride=(1, 1), alpha=alpha, id=11)

    # Conv dw / s2: filter shape (3 x 3 x 512 dw)
    x = depthwise_block(x, kernel=(3, 3), stride=(2, 2), depth_multiplier=depth_multiplier, id=12)
    # Conv pw / s1: filter shape (1 x 1 x 512 x 1024)
    x = pointwise_block(x, num_filters=1024, kernel=(1, 1), stride=(1, 1), alpha=alpha, id=12)

    # Conv dw / s2: filter shape (3 x 3 x 1024 dw)
    x = depthwise_block(x, kernel=(3, 3), stride=(1, 1), depth_multiplier=depth_multiplier, id=13)
    # Conv pw / s1: filter shape (1 x 1 x 1024 x 1024)
    x = pointwise_block(x, num_filters=1024, kernel=(1, 1), stride=(1, 1), alpha=alpha, id=13)

    # average pooling
    if include_top:
        raise NotImplementedError
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # create model
    rows = input_shape[0]
    model = models.Model(inputs, x, name='mobilenet_%0.2f_%s' % (alpha, rows))

    # load pre-trained weights on ImageNet
    if weights == 'imagenet':
        if alpha == 1.0:
            alpha_text = '1_0'
        elif alpha == 0.75:
            alpha_text = '7_5'
        elif alpha == 0.50:
            alpha_text = '5_0'
        else:
            alpha_text = '2_5'

        BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
                            'releases/download/v0.6/')
        if include_top:
            model_name = 'mobilenet_%s_%d_tf.h5' % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = keras.utils.get_file(model_name,
                                                weight_path,
                                                cache_subdir='models')
        else:
            model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = keras.utils.get_file(model_name,
                                                weight_path,
                                                cache_subdir='models')
        model.load_weights(weights_path)
    else:
        raise NotImplementedError

    return model


def depthwise_block(inputs, kernel, stride, depth_multiplier, id):
    x = layers.ZeroPadding2D(padding=(1, 1), name='convdw_zeropad_%d' % id)(inputs)
    x = layers.DepthwiseConv2D(kernel_size=kernel, padding='valid', depth_multiplier=depth_multiplier, strides=stride,
                               use_bias=False, name='convdw_%d' % id)(x)
    x = layers.BatchNormalization(axis=-1, name='convdw_bn_%d' % id)(x)
    return layers.Activation(relu6, name='convdw_relu6_%d' % id)(x)


def pointwise_block(inputs, num_filters, kernel, stride, alpha, id):
    num_filter = int(alpha * num_filters)
    x = layers.Conv2D(filters=num_filter, padding='same', kernel_size=kernel, strides=stride, use_bias=False,
                      name='convpw_%d' % id)(inputs)
    x = layers.BatchNormalization(axis=-1, name='convpw_bn_%d' % id)(x)
    return layers.Activation(relu6, name='convpw_relu6_%d' % id)(x)
