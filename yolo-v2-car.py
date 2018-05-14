import keras.backend as K
import numpy as np
import scipy.io
import scipy.misc
import os
import tensorflow as tf
from matplotlib.pyplot import imshow
from utils import yolo_boxes_to_corners, scale_boxes, read_classes, \
    read_anchors, preprocess_image, generate_colors, draw_boxes
from keras.models import load_model
from yad2k.models.keras_yolo import yolo_head


# Two-step filtering: Step 1
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """ get rid of any box for which the class score is less than a chosen threshold, which means
    the box is not very confident about detecting a class
    :param box_confidence: tensor of shape (19*19,5,1) Pc, confidence probability that
                        there's some objectnfor each anchor box (5 in total in our case).
    :param boxes: tensor of shape (19*19,5,4) containing (bx,by,bh,bw) for each of the 5 boxes per cell.
    :param box_class_probs: tensor of shape (19*19,5,80) containing the detection prob (c1,...,c80) for
                        each of the 80 classes for each of the 5 boxes per cell.
    :param threshold: remove the box which highest class prob score < threshold
    :return:
    """
    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)  # return the index by argmax taken from the last axis
    box_class_scores = K.max(box_scores, axis=-1)

    # Create a mask by using the threshold
    filtering_mask = box_class_scores >= threshold

    # Apply the mask to scores, boxes and classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes


# Two-step filtering: Step 2 (Non-max suppression)
def IoU(box1, box2):
    """ Calculate the Intersection over Union between box1 and box2
    :param box1: a box with coordinates (x1, y1, x2, y2)
    :param box2: another box with coordinates (x1, y1, x2, y2)
    """
    # Calculate the intersection area with two corners: lower left and upper right
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])
    inter_area = max((xi2 - xi1), 0) * max((yi2 - yi1), 0)

    # Calculate Union area by using area1 + area2 - Inter(1,2)
    box1_area = np.float((np.maximum(box1[0], box1[2]) - np.minimum(box1[0], box1[2])) * (
            np.maximum(box1[1], box1[3]) - np.minimum(box1[1], box1[3])))
    box2_area = np.float((np.maximum(box2[0], box2[2]) - np.minimum(box2[0], box2[2])) * (
            np.maximum(box2[1], box2[3]) - np.minimum(box2[1], box2[3])))
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area

    return iou


def non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """ Apply non-max suppression to remove overlapped boxes based iou_threshold
    :param scores: output of yolo_filter_boxes
    :param boxes: output of yolo_filter_boxes
    :param classes: output of yolo_filter_boxes
    :param max_boxes: maximum number of predicted boxes you'd like
    :param iou_threshold: used for NMS filtering

    :return: return scores, boxes, classes after NMS filtering (second time filtering)
    """
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

    # Prune away boxes that have high IoU overlap with previously selected boxes
    selected_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)

    # Select scores, boxes, classes which only match with selected_indices from last step
    scores = tf.gather(scores, selected_indices)
    boxes = tf.gather(boxes, selected_indices)
    classes = tf.gather(classes, selected_indices)

    return scores, boxes, classes


def yolo_eval(yolo_ouputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    :param yolo_ouputs:
    :param image_shape:
    :param max_boxes:
    :param score_threshold:
    :param iou_threshold:
    :return:
    """
    box_confidence, box_xy, box_wh, box_class_probs = yolo_ouputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # filter boxes with low prob of having an object - scale - NMS
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
    boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes


def predict(sess, image_file):
    image, image_data = preprocess_image('./images/' + image_file, model_image_size=(608, 608))
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input: image_data,
                                                             K.learning_phase(): 0})
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    colors = generate_colors(class_names)
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    image.save(os.path.join("out", image_file), quality=90)
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)

    return out_scores, out_boxes, out_classes


if __name__ == '__main__':
    # Necessary model data and pre-trained yolo_model with parameters from YOLO official website
    sess = K.get_session()
    class_names = read_classes('./coco_classes.txt')
    anchors = read_anchors('./yolo_anchors.txt')
    image_shape = (720., 1280.)

    yolo_model = load_model('./yolo.h5')
    yolo_model.summary()

    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
    out_scores, out_boxes, out_classes = predict(sess, 'test.jpg')

    print('out_scores: {}. out_boxes: {}. out_classes: {}.'.format(out_scores, out_boxes, out_classes))



