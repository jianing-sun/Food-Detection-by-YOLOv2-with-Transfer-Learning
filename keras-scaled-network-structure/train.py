#! /usr/bin/env python

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam

from callbacks import CustomModelCheckpoint, CustomTensorBoard
from generator import BatchGenerator
from utils.multi_gpu_model import multi_gpu_model
from utils.utils import normalize, evaluate, makedirs
from voc import parse_annotation
from yolo import create_yolov3_model, dummy_loss


def create_train_valid_set(
        train_annot_folder,
        train_image_folder,
        valid_annot_folder,
        valid_image_folder,
        labels,
):
    # parse annotations for all images
    all_images = []
    all_seen_labels = []
    for i in range(0, len(labels)):
        image_path = train_image_folder + str(i + 1) + '/'
        annot_path = train_annot_folder + str(i + 1) + '/' + 'annotations_new/'

        train_ints, seen_labels = parse_annotation(annot_path, image_path, labels)

        all_images.extend(train_ints)
        for seen_label, counts in seen_labels.items():
            if seen_label not in all_seen_labels:
                all_seen_labels.append(seen_label)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_annotation(valid_annot_folder, valid_image_folder, labels)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")
        train_valid_split = int(0.8 * len(all_images))
        np.random.seed(0)
        np.random.shuffle(all_images)
        np.random.seed()

        valid_ints = all_images[train_valid_split:]
        train_ints = all_images[:train_valid_split]

    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(all_seen_labels))
        if len(overlap_labels) == len(labels) == len(all_seen_labels):
            print('All 100 categories appeared.\n')

    max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])
    print('Maximum box number per image: ' + str(max_box_per_image) + '\n')

    return train_ints, valid_ints, sorted(labels), max_box_per_image


def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save):
    makedirs(tensorboard_logs)

    early_stop = EarlyStopping(
        monitor='loss',
        min_delta=0.01,
        patience=5,
        mode='min',
        verbose=1
    )
    checkpoint = CustomModelCheckpoint(
        model_to_save=model_to_save,
        filepath=saved_weights_name,  # + '{epoch:02d}.h5',
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='min',
        epsilon=0.01,
        cooldown=0,
        min_lr=0
    )
    tensorboard = CustomTensorBoard(
        log_dir=tensorboard_logs,
        write_graph=True,
        write_images=True,
    )
    return [early_stop, checkpoint, reduce_on_plateau, tensorboard]


def create_model(
        nb_class,
        anchors,
        max_box_per_image,
        max_grid, batch_size,
        warmup_batches,
        ignore_thresh,
        multi_gpu,
        saved_weights_name,
        lr,
        grid_scales,
        obj_scale,
        noobj_scale,
        xywh_scale,
        class_scale
):
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            template_model, infer_model = create_yolov3_model(
                nb_class=nb_class,
                anchors=anchors,
                max_box_per_image=max_box_per_image,
                max_grid=max_grid,
                batch_size=batch_size // multi_gpu,
                warmup_batches=warmup_batches,
                ignore_thresh=ignore_thresh,
                grid_scales=grid_scales,
                obj_scale=obj_scale,
                noobj_scale=noobj_scale,
                xywh_scale=xywh_scale,
                class_scale=class_scale
            )
    else:
        template_model, infer_model = create_yolov3_model(
            nb_class=nb_class,
            anchors=anchors,
            max_box_per_image=max_box_per_image,
            max_grid=max_grid,
            batch_size=batch_size,
            warmup_batches=warmup_batches,
            ignore_thresh=ignore_thresh,
            grid_scales=grid_scales,
            obj_scale=obj_scale,
            noobj_scale=noobj_scale,
            xywh_scale=xywh_scale,
            class_scale=class_scale
        )

    # load the pretrained weight if exists, otherwise load the backend weight only
    # if os.path.exists(saved_weights_name):
    #     print("\nLoading pretrained weights.\n")
    #     template_model.load_weights(saved_weights_name)
    # else:
    #     template_model.load_weights("backend.h5", by_name=True)

    if multi_gpu > 1:
        train_model = multi_gpu_model(template_model, gpus=multi_gpu)
    else:
        train_model = template_model

    optimizer = Adam(lr=lr, clipnorm=0.001)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)

    return train_model, infer_model


def read_category():
    category = []
    with open('/Volumes/JS/UECFOOD100_JS/category.txt', 'r') as file:
        for i, line in enumerate(file):
            if i > 0:
                line = line.rstrip('\n')
                line = line.split('\t')
                category.append(line[1])
    return category


def _main_():
    config_path = './config.json'
    LABELS = read_category()

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ''' Parse annotations '''
    config['model']['labels'] = LABELS
    train_ints, valid_ints, labels, max_box_per_image = create_train_valid_set(
        config['train']['train_annot_folder'],
        config['train']['train_image_folder'],
        config['valid']['valid_annot_folder'],
        config['valid']['valid_image_folder'],
        config['model']['labels']
    )

    ''' Create generators '''
    train_generator = BatchGenerator(
        instances=train_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        # min_net_size=config['model']['min_input_size'],
        # max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=True,
        # norm=normalize
    )

    # used to check if images are normal after BatchGenerator
    img = train_generator[0][0][0][5]
    plt.imshow(img.astype('uint8'))

    valid_generator = BatchGenerator(
        instances=valid_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        # min_net_size=config['model']['min_input_size'],
        # max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=False,
        # norm=normalize
    )

    ''' Create the model '''
    if os.path.exists(config['train']['saved_weights_name']):
        config['train']['warmup_epochs'] = 0
    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times'] * len(train_generator))

    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    multi_gpu = len(config['train']['gpus'].split(','))

    train_model, infer_model = create_model(
        nb_class=len(labels),
        anchors=config['model']['anchors'],
        max_box_per_image=max_box_per_image,
        max_grid=[config['model']['max_input_size'], config['model']['max_input_size']],
        batch_size=config['train']['batch_size'],
        warmup_batches=warmup_batches,
        ignore_thresh=config['train']['ignore_thresh'],
        multi_gpu=multi_gpu,
        saved_weights_name=config['train']['saved_weights_name'],
        lr=config['train']['learning_rate'],
        grid_scales=config['train']['grid_scales'],
        obj_scale=config['train']['obj_scale'],
        noobj_scale=config['train']['noobj_scale'],
        xywh_scale=config['train']['xywh_scale'],
        class_scale=config['train']['class_scale'],
    )

    ''' Kick off the training '''
    callbacks = create_callbacks(config['train']['saved_weights_name'], config['train']['tensorboard_dir'], infer_model)
    train_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator) * config['train']['train_times'],
        epochs=config['train']['nb_epochs'] + config['train']['warmup_epochs'],
        verbose=2 if config['train']['debug'] else 1,
        callbacks=callbacks,
        workers=4,
        max_queue_size=8
    )

    # make a GPU version of infer_model for evaluation
    if multi_gpu > 1:
        infer_model = load_model(config['train']['saved_weights_name'])

    ''' Evaluate the result '''
    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))


if __name__ == '__main__':
    _main_()
