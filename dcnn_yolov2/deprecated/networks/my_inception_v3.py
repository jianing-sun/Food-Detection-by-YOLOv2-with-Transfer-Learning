import os

from keras_applications import get_keras_submodule
from keras_applications.imagenet_utils import _obtain_input_shape

backend = get_keras_submodule('backend')
engine = get_keras_submodule('engine')
layers = get_keras_submodule('layers')
models = get_keras_submodule('models')
keras_utils = get_keras_submodule('utils')

WEIGHTS_PATH = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.5/'
    'inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.5/'
    'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    if padding == 'valid':
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x


def myInceptionV3(include_top=True,
                  weights='imagenet',
                  input_tensor=None,
                  input_shape=None,
                  pooling=None,
                  classes=1000):
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=backend.image_data_format(),
        require_flatten=False,
        weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    # x = layers.UpSampling2D(2)(img_input)
    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3, padding='valid')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1, padding='same')

    branch5x5 = conv2d_bn(x, 48, 1, 1, padding='same')
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, padding='same')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, padding='same')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, padding='same')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, padding='same')

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1, padding='same')
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 56 x 56 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1, padding='same')

    branch5x5 = conv2d_bn(x, 48, 1, 1, padding='same')
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, padding='same')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, padding='same')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, padding='same')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, padding='same')

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, padding='same')
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 56 x 56 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1, padding='same')

    branch5x5 = conv2d_bn(x, 48, 1, 1, padding='same')
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, padding='same')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, padding='same')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, padding='same')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, padding='same')

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, padding='same')
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3:
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, padding='same')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')  # 28 x 28

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, padding='same')

    branch7x7 = conv2d_bn(x, 128, 1, 1, padding='same')
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7, padding='same')
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, padding='same')

    branch7x7dbl = conv2d_bn(x, 128, 1, 1, padding='same')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, padding='same')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7, padding='same')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, padding='same')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, padding='same')

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, padding='same')
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1, padding='same')

        branch7x7 = conv2d_bn(x, 160, 1, 1, padding='same')
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7, padding='same')
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, padding='same')

        branch7x7dbl = conv2d_bn(x, 160, 1, 1, padding='same')
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1, padding='same')
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7, padding='same')
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1, padding='same')
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, padding='same')

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, padding='same')
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, padding='same')

    branch7x7 = conv2d_bn(x, 192, 1, 1, padding='same')
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7, padding='same')
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, padding='same')

    branch7x7dbl = conv2d_bn(x, 192, 1, 1, padding='same')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1, padding='same')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, padding='same')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1, padding='same')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, padding='same')

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, padding='same')
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')  # 28x28

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1, padding='same')
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1, padding='same')
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7, padding='same')
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1, padding='same')
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')  # 14x14

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1, padding='same')

        branch3x3 = conv2d_bn(x, 384, 1, 1, padding='same')
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3, padding='same')
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1, padding='same')
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1, padding='same')
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3, padding='same')
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3, padding='same')
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1, padding='same')
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, padding='same')
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = engine.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='inception_v3')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
        else:
            weights_path = keras_utils.get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='bcbd6486424b2319ff4ef7d526e38f63')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model
