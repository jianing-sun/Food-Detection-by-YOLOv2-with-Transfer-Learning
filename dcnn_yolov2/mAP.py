import cv2
import numpy as np
import os
from utils import BoundBox, bbox_iou, decode_netout, read_category
from scipy.special import expit
import matplotlib.pyplot as plt
from keras_applications.mobilenet_v2 import MobileNetV2
from matplotlib.patches import Rectangle


def _sigmoid(x):
    return expit(x)


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

    return boxes


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue

            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
    return boxes


def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) // new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) // new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:, :, ::-1] / 255., (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h - new_h) // 2:(net_h + new_h) // 2, (net_w - new_w) // 2:(net_w + new_w) // 2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image


def normalize(image):
    return image / 255.


def get_yolo_boxes(idx, model, images, net_h, net_w, anchors, obj_thresh, nms_thresh, labels):

    image_h, image_w, _ = images[0].shape
    nb_images = len(images)
    batch_input = np.zeros((nb_images, net_h, net_w, 3))

    # preprocess the input
    # for i in range(nb_images):
    #     batch_input[i] = preprocess_input(images[i], net_h, net_w)

    for i in range(nb_images):
        batch_input[i] = images[i]

    # run the prediction
    # plt.imshow(batch_input[0].astype('float'))

    TRUE_BOX_BUFFER = 15
    dummy_array = np.zeros((1, 1, 1, 1, TRUE_BOX_BUFFER, 4))

    batch_output = model.predict_on_batch([batch_input, dummy_array])
    batch_boxes = [None] * nb_images

    for i in range(nb_images):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(batch_input[i])

        # remove axes
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # yolos = [batch_output[0][i], batch_output[1][i], batch_output[2][i], batch_output[3][i], batch_output[4][i]]
        boxes = []

        # decode the output of the network
        # for j in range(len(yolos)):
        #     yolo_anchors = anchors[:]  # config['model']['anchors']
        #     boxes += decode_netout(yolos[j], yolo_anchors, obj_thresh, net_h, net_w)

        # decode the output of the network
        yolo_anchors = anchors
        # yolo_anchors = [int(b*32) for b in anchors]
        boxes += decode_netout(netout=batch_output[i],
                               anchors=yolo_anchors,
                               obj_threshold=obj_thresh)

        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)

        batch_boxes[i] = boxes

        if boxes is not None:
            for box in boxes:
                xmin = int(box.xmin)
                ymin = int(box.ymin)
                xmax = int(box.xmax)
                ymax = int(box.ymax)

                ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       facecolor='none', edgecolor='green', linewidth=3.0))
                ax.text(xmin, ymax, labels[box.get_label()] + ' ' + str('{0:.3f}'.format(box.get_conf())),
                        backgroundcolor='limegreen', alpha=0.5)

        # image = draw_boxes(batch_input[i], ax, boxes, labels=labels)
        else:
            pass

        result_path = '/Volumes/JS/Result_uecfood100/mn_normal_Ocb16/'
        fig.savefig(result_path + str(idx) + '_%s' % str(i) + '.png')

    return batch_boxes


def draw_boxes(image, ax, boxes, labels):
    image_h, image_w, _ = image.shape

    for box in boxes:
        xmin = int(box.xmin)
        ymin = int(box.ymin)
        xmax = int(box.xmax)
        ymax = int(box.ymax)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
        cv2.putText(image,
                    labels[box.get_label()] + ' ' + str(box.get_score()),
                    (xmin, ymin - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * image_h,
                    (255, 255, 255), 1)

        ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                               facecolor='none', edgecolor='blue', linewidth=3.0))
        ax.text(xmin+20, ymin+50, labels[box.get_label()] + ' ' + str(box.get_score()), backgroundcolor='cornflowerblue')

    return image


def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate(model,
             generator,
             iou_threshold=0.5):
    """ Evaluate a given dataset using a given model.
    code originally from https://github.com/fizyr/keras-retinanet
    # Arguments
        model           : The model to evaluate.
        generator       : The generator that represents the dataset to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    labels = read_category()

    for i in range(generator.size()):
        raw_image = generator.load_image(i)

        raw_height, raw_width, raw_channels = raw_image.shape

        # make the boxes and the labels
        pred_boxes = predict(i, raw_image, model, labels)

        score = np.array([box.score for box in pred_boxes])
        pred_labels = np.array([box.label for box in pred_boxes])

        if len(pred_boxes) > 0:
            pred_boxes = np.array([[min(int(box.xmin * raw_width), raw_width),
                                    min(int(box.ymin * raw_height), raw_height),
                                    min(int(box.xmax * raw_width), raw_width),
                                    min(int(box.ymax * raw_height), raw_height),
                                    box.score] for box in pred_boxes])
        else:
            pred_boxes = np.array([[]])

        # sort the boxes and the labels according to scores
        score_sort = np.argsort(-score)
        pred_labels = pred_labels[score_sort]
        pred_boxes = pred_boxes[score_sort]

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = pred_boxes[pred_labels == label, :]

        annotations = generator.load_annotation(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

    # compute mAP by comparing all detections and all annotations
    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision

    return average_precisions


def _softmax(x, axis=-1):
    x = x - np.amax(x, axis, keepdims=True)
    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)


def predict(idx, image, model, labels):
    image_h, image_w, _ = image.shape
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(image[:, :, ::-1])

    image = cv2.resize(image, (224, 224))
    image = image / 255.

    input_image = image[:, :, ::-1]
    input_image = np.expand_dims(input_image, 0)
    dummy_array = np.zeros((1, 1, 1, 1, 50, 4))

    netout = model.predict([input_image, dummy_array])[0]

    anchors = [4.33, 3.64, 6.92, 6.26, 10.81, 7.48, 10.81, 4.86, 12.20, 9.29]
    # anchors = [1.91, 1.61, 3.53, 2.97, 5.04, 4.38, 6.20, 3.33, 6.67, 4.90]

    # uecfood256
    # anchors = [2.27, 1.93, 3.87, 3.03, 5.27, 4.30, 6.23, 3.29, 6.56, 4.77]

    boxes = decode_netout(netout, anchors, 100)

    # box_colors = ['#FF0000', '#FFFF00', '#00FF00', '#0000FF', '#00FFFF', '#FF00FF',
    #               '#FFA500', '#FF3700', '#800080', '#00FF19']
    #
    # # # remove axes
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #
    # if boxes is not None:
    #     for i, box in enumerate(boxes):
    #         xmin = min(int(box.xmin * image_w), image_w)
    #         ymin = min(int(box.ymin * image_h), image_h)
    #         xmax = min(int(box.xmax * image_w), image_w)
    #         ymax = min(int(box.ymax * image_h), image_h)
    #
    #         ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                facecolor='none', edgecolor=box_colors[i], linewidth=3.0))
    #         ax.text(xmin, ymax, labels[box.get_label()] + ' ' + str('{0:.3f}'.format(box.get_conf())),
    #                 backgroundcolor=box_colors[i], alpha=1)
    # # #
    # result_path = '/Volumes/JS/Result_uecfood100/mn2_tla_256/'
    # fig.savefig(result_path + str(idx) + '.png')

    return boxes