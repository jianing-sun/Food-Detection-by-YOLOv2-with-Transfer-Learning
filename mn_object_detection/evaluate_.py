import numpy as np
from preprocessing import parse_annotation, BatchGenerator
from Evaluate import evaluate
import matplotlib.pyplot as plt

from keras.models import load_model, Model
from keras.applications.mobilenet import MobileNet
from keras.layers import Reshape, Conv2D, Input, Lambda
from keras_applications.mobilenet import _depthwise_conv_block


def normalize(image):
    return image / 255.


def read_category():
    category = []
    with open('/Volumes/JS/UECFOOD100_JS/category.txt', 'r') as file:
        for i, line in enumerate(file):
            if i > 0:
                line = line.rstrip('\n')
                line = line.split('\t')
                category.append(line[1])
    return category


def get_normal_mn1():
    print('=> Building MobileNetV1 model...')
    mobilenet = MobileNet(input_shape=(224, 224, 3), include_top=False)
    x = mobilenet(input_image)
    x = Conv2D(N_BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
    output = Reshape((GRID_H, GRID_W, N_BOX, 4 + 1 + CLASS))(x)
    output = Lambda(lambda args: args[0])([output, true_boxes])

    model = Model([input_image, true_boxes], output)
    print(model.summary())
    return model


if __name__ == '__main__':
    ''' Initiailize parameters '''
    LABELS = read_category()

    IMAGE_H, IMAGE_W = 224, 224  # must equal to GRID_H * 32  416, 416
    GRID_H, GRID_W = 7, 7  # 13, 13
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
        image_path = '/Volumes/JS/UECFOOD100_JS/' + str(i + 1) + '/'
        annot_path = '/Volumes/JS/UECFOOD100_JS/' + str(i + 1) + '/' + '/annotations_new/'

        folder_imgs, seen_labels = parse_annotation(annot_path, image_path)
        all_imgs.extend(folder_imgs)
    print(np.array(all_imgs).shape)

    # add extensions to image name
    for img in all_imgs:
        img['filename'] = img['filename']

    print('=> Generate BatchGenerator.')
    batches = BatchGenerator(all_imgs, generator_config)

    train_valid_split = int(0.8 * len(all_imgs))

    train_batch = BatchGenerator(all_imgs[:train_valid_split], generator_config, norm=normalize, jitter=False)
    valid_batch = BatchGenerator(all_imgs[train_valid_split:], generator_config, norm=normalize, jitter=False)

    input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
    true_boxes = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER, 4))

    model = get_normal_mn1()
    model.load_weights('./record/mnv2_224_1007_normal/mnv2224_normal_1007_gcp.h5')

    average_precisions = evaluate(model, valid_batch, obj_thresh=0.3, net_h=224, net_w=224)