import os
import cv2
from keras.applications import inception_v3
from keras_applications.mobilenet import MobileNet

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import Reshape, Conv2D, Input, Lambda, UpSampling2D, MaxPooling2D, LeakyReLU, BatchNormalization
from keras import layers
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras_applications.mobilenet import _depthwise_conv_block
from keras_applications.mobilenet_v2 import _inverted_res_block

from networks.my_inception_v3 import myInceptionV3
from preprocessing import parse_annotation, BatchGenerator


def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)


def normalize(image):
    return image / 255.


def add_mn2(x, alpha=1.0):
    x = UpSampling2D(2)(x)  # 56x56
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                            expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=9)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=12)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                            expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=15)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                            expansion=6, block_id=16)

    return x


def add_mn(x):
    alpha, depth_multiplier = 1, 1
    x = UpSampling2D(2)(x)    # 56x56
    x = _depthwise_conv_block(x, 768, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(x, 768, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 768, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 768, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 768, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 768, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 768, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 768, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 768, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
    return x


def get_model():
    """ Build MobileNetV1 model """
    print('=> Building MobileNetV1 model...')
    small_incepv3 = myInceptionV3(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    # print('small_incepv3.summary()', small_incepv3.summary())
    small_incepv3 = Model(inputs=small_incepv3.inputs, outputs=small_incepv3.layers[-84].input)
    # print(small_incepv3.summary())
    x = small_incepv3(input_image)
    x = add_mn(x)
    x = Conv2D(N_BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
    output = Reshape((GRID_H, GRID_W, N_BOX, 4 + 1 + CLASS))(x)

    # small hack to allow true_boxes to be registered when Keras build the model
    # for more information: https://github.com/fchollet/keras/issues/2790
    output = Lambda(lambda args: args[0])([output, true_boxes])

    model = Model([input_image, true_boxes], output)
    print(model.summary())
    return model


def addDarknet(x):
    # Layer 13    28x28
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_dn_13', use_bias=False)(x)
    x = BatchNormalization(name='norm_13')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14   14x14
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_dn_14', use_bias=False)(x)
    x = BatchNormalization(name='norm_14')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_dn_15', use_bias=False)(x)
    x = BatchNormalization(name='norm_15')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_dn_16', use_bias=False)(x)
    x = BatchNormalization(name='norm_16')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 17   14x14
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_dn_17', use_bias=False)(x)
    x = BatchNormalization(name='norm_17')(x)
    x = LeakyReLU(alpha=0.1)(x)

    skip_connection = x

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 18   7x7
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_dn_18', use_bias=False)(x)
    x = BatchNormalization(name='norm_18')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
    x = BatchNormalization(name='norm_19')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
    x = BatchNormalization(name='norm_20')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 21
    skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(
        skip_connection)
    skip_connection = BatchNormalization(name='norm_21')(skip_connection)
    skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
    skip_connection = Lambda(space_to_depth_x2)(skip_connection)

    x = concatenate([skip_connection, x])

    # Layer 22
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
    x = BatchNormalization(name='norm_22')(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def get_pretrained_mn1():
    alpha, depth_multiplier = 1, 1
    print('=> Building new model with pretrained MobilenetV1...')

    pretrained_gap_model = load_model('./models/gap_foodnotfood_mn1_224.h5')
    print(pretrained_gap_model.summary())
    model = Model(inputs=pretrained_gap_model.input, outputs=pretrained_gap_model.layers[-6].input)
    print(model.summary())
    x = model(input_image)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    x = Conv2D(N_BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
    output = Reshape((GRID_H, GRID_W, N_BOX, 4 + 1 + CLASS))(x)
    output = Lambda(lambda args: args[0])([output, true_boxes])

    model = Model([input_image, true_boxes], output)
    print(model.summary())
    print('Finish new model.')
    return model


def get_darknet_with_myIncepv3():
    """ Build MobileNetV1 model """
    print('=> Building MobileNetV1 model...')
    small_incepv3 = myInceptionV3(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    # print('small_incepv3.summary()', small_incepv3.summary())
    small_incepv3 = Model(inputs=small_incepv3.inputs, outputs=small_incepv3.layers[-84].input)
    print(small_incepv3.summary())
    x = small_incepv3(input_image)
    x = addDarknet(x)
    x = Conv2D(N_BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
    output = Reshape((GRID_H, GRID_W, N_BOX, 4 + 1 + CLASS))(x)

    # small hack to allow true_boxes to be registered when Keras build the model
    # for more information: https://github.com/fchollet/keras/issues/2790
    output = Lambda(lambda args: args[0])([output, true_boxes])

    model = Model([input_image, true_boxes], output)
    print(model.summary())
    return model


def train(model):

    layer = model.layers[-4]            # the last convolutional layer
    weights = layer.get_weights()

    new_kernel = np.random.normal(size=weights[0].shape) / (GRID_H * GRID_W)
    new_bias = np.random.normal(size=weights[1].shape) / (GRID_H * GRID_W)

    layer.set_weights([new_kernel, new_bias])

    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.001,
                               patience=3,
                               mode='min',
                               verbose=1)

    checkpoint = ModelCheckpoint('dlp_tla_mn224_e1_1001.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 period=1)

    reduce_lr = ReduceLROnPlateau(patience=0, factor=0.2, monitor='val_loss', verbose=1)

    # model.load_weights('./transferLearning_mn_224_0_04425.h5')

    tb_counter = len([log for log in os.listdir(os.path.expanduser('./tl_tf_logs/')) if 'uecfood100' in log]) + 1
    tensorboard = TensorBoard(log_dir=os.path.expanduser('~/tf_log/') + 'transferLearning_uecfood100' + '_' + str(tb_counter),
                              histogram_freq=0,
                              write_graph=True,
                              write_images=False)

    # TODO: try different optimizer and tweak parameters (in MNv1 paper they used RMSprop)
    optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
    # optimizer = RMSprop(lr=1e-5, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(loss=custom_loss, optimizer=optimizer)

    hist = model.fit_generator(generator=train_batch,
                               steps_per_epoch=len(train_batch),
                               epochs=20,      # 100
                               verbose=1,
                               validation_data=valid_batch,
                               validation_steps=len(valid_batch),
                               callbacks=[early_stop, checkpoint, tensorboard, reduce_lr],
                               max_queue_size=3)

    loss_hist = hist.history['loss']
    np_loss_hist = np.array(loss_hist)
    np.savetxt('loss_history_dlp_tla_mn224_e20_1001.txt', np_loss_hist, delimiter=',')


def custom_loss(y_true, y_pred):
    mask_shape = tf.shape(y_true)[:4]

    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)))
    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [BATCH_SIZE, 1, 1, 5, 1])

    coord_mask = tf.zeros(mask_shape)
    conf_mask = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)

    seen = tf.Variable(0.)
    total_recall = tf.Variable(0.)

    """ Adjust prediction """
    # adjust x and y
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

    # adjust w and h
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1, 1, 1, N_BOX, 2])

    # adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])

    # adjust class probabilities
    pred_box_class = y_pred[..., 5:]

    """ Adjust ground truth """
    # adjust x and y
    true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

    # adjust w and h
    true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

    # adjust confidence
    true_wh_half = true_box_wh / 2.
    true_mins = true_box_xy - true_wh_half
    true_maxes = true_box_xy + true_wh_half

    pred_wh_half = pred_box_wh / 2.
    pred_mins = pred_box_xy - pred_wh_half
    pred_maxes = pred_box_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    true_box_conf = iou_scores * y_true[..., 4]

    # adjust class probabilities
    true_box_class = tf.argmax(y_true[..., 5:], -1)

    """ Determine the masks """
    # coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE

    # confidence mask: penelize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE

    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE

    # class mask: simply the position of the ground truth boxes (the predictors)
    class_mask = y_true[..., 4] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE

    """ Warm-up training """
    no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE / 2.)
    seen = tf.assign_add(seen, 1.)

    true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES),
                                                   lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                                            true_box_wh + tf.ones_like(true_box_wh) * np.reshape(
                                                                ANCHORS, [1, 1, 1, N_BOX, 2]) * no_boxes_mask,
                                                            tf.ones_like(coord_mask)],
                                                   lambda: [true_box_xy,
                                                            true_box_wh,
                                                            coord_mask])

    """ Finalize the loss """
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

    loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

    loss = loss_xy + loss_wh + loss_conf + loss_class

    nb_true_box = tf.reduce_sum(y_true[..., 4])
    nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

    """ Debugging code """
    current_recall = nb_pred_box / (nb_true_box + 1e-6)
    total_recall = tf.assign_add(total_recall, current_recall)

    loss = tf.Print(loss, [tf.zeros((1))], message='\nDummy Line \t', summarize=1000)
    loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)

    return loss


def read_category():
    category = []
    with open('/Volumes/JS/UECFOOD100_JS/category.txt', 'r') as file:
        for i, line in enumerate(file):
            if i > 0:
                line = line.rstrip('\n')
                line = line.split('\t')
                category.append(line[1])
    return category


def plt_example_batch(batches, batch_size=16):
    assert batches[0][0][0].shape[0] == batch_size       # in general 16x224x224x3
    for i in range(0, batch_size):
        img = batches[0][0][0][i]
        plt.figure(i)
        plt.imshow(img.astype('uint8'))


if __name__ == '__main__':

    ''' Initiailize parameters '''
    LABELS = read_category()

    IMAGE_H, IMAGE_W = 224, 224  # must equal to GRID_H * 32  416, 416
    GRID_H, GRID_W = 7, 7        # 13, 13
    N_BOX = 5
    CLASS = len(LABELS)
    CLASS_WEIGHTS = np.ones(CLASS, dtype='float32')
    OBJ_THRESHOLD = 0.3
    NMS_THRESHOLD = 0.3

    # Read knn generated anchor_5.txt
    ANCHORS = []
    with open('/Volumes/JS/UECFOOD100_JS/generated_anchors_mobilenet/anchors_5.txt', 'r') as anchor_file:
        for i, line in enumerate(anchor_file):
            line = line.rstrip('\n')
            ANCHORS.append(list(map(float, line.split(', '))))
    ANCHORS = list(list(np.array(ANCHORS).reshape(1, -1))[0])

    NO_OBJECT_SCALE = 1.0
    OBJECT_SCALE = 5.0
    COORD_SCALE = 1.0
    CLASS_SCALE = 1.0

    BATCH_SIZE = 16
    WARM_UP_BATCHES = 100
    TRUE_BOX_BUFFER = 15

    generator_config = {
        'IMAGE_H': IMAGE_H,
        'IMAGE_W': IMAGE_W,
        'GRID_H': GRID_H,
        'GRID_W': GRID_W,
        'BOX': N_BOX,
        'LABELS': LABELS,
        'CLASS': len(LABELS),
        'ANCHORS': ANCHORS,
        'BATCH_SIZE': BATCH_SIZE,
        'TRUE_BOX_BUFFER': TRUE_BOX_BUFFER,
    }

    all_imgs = []
    for i in range(0, len(LABELS)):
        image_path = '/Volumes/JS/UECFOOD100_JS/' + str(i+1) + '/'
        annot_path = '/Volumes/JS/UECFOOD100_JS/' + str(i+1) + '/' + '/annotations_new/'

        folder_imgs, seen_labels = parse_annotation(annot_path, image_path)
        all_imgs.extend(folder_imgs)
    print(np.array(all_imgs).shape)

    # add extensions to image name
    for img in all_imgs:
        img['filename'] = img['filename']

    print('=> Generate BatchGenerator.')
    batches = BatchGenerator(all_imgs, generator_config)

    img = batches[0][0][0][5]
    plt.imshow(img.astype('uint8'))
    # plt_example_batch(batches, BATCH_SIZE)

    ''' Start training '''
    train_valid_split = int(0.8 * len(all_imgs))

    train_batch = BatchGenerator(all_imgs[:train_valid_split], generator_config, norm=normalize, jitter=False)
    valid_batch = BatchGenerator(all_imgs[train_valid_split:], generator_config, norm=normalize, jitter=False)

    input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
    true_boxes = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER, 4))

    model = get_pretrained_mn1()
    # model = get_darknet_with_myIncepv3()
    # model = get_model()

    train(model)
