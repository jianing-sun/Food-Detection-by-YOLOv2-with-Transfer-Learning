import keras.backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from keras.layers import Conv2D, Input, BatchNormalization, ZeroPadding2D, \
    UpSampling2D, DepthwiseConv2D, Activation, concatenate
from keras.layers.merge import add
from keras.models import Model


class YoloLayer(Layer):
    def __init__(self, anchors, max_grid, batch_size, warmup_batches, ignore_thresh,
                 grid_scale, obj_scale, noobj_scale, xywh_scale, class_scale,
                 **kwargs):
        # make the model settings persistent
        self.ignore_thresh = ignore_thresh
        self.warmup_batches = warmup_batches
        self.anchors = tf.constant(anchors, dtype='float', shape=[1, 1, 1, 3, 2])
        self.grid_scale = grid_scale
        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        self.xywh_scale = xywh_scale
        self.class_scale = class_scale

        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
        self.cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, 3, 1])

        super(YoloLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(YoloLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        input_image, y_pred, y_true, true_boxes = x

        # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
        y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))

        # initialize the masks
        object_mask = tf.expand_dims(y_true[..., 4], 4)

        # the variable to keep track of number of batches processed
        batch_seen = tf.Variable(0.)

        # compute grid factor and net factor
        grid_h = tf.shape(y_true)[1]
        grid_w = tf.shape(y_true)[2]
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 2])

        net_h = tf.shape(input_image)[1]
        net_w = tf.shape(input_image)[2]
        net_factor = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1, 1, 1, 1, 2])

        """
        Adjust prediction
        """
        pred_box_xy = (self.cell_grid[:, :grid_h, :grid_w, :, :] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy
        pred_box_wh = y_pred[..., 2:4]  # t_wh
        pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)  # adjust confidence
        pred_box_class = y_pred[..., 5:]  # adjust class probabilities

        """
        Adjust ground truth
        """
        true_box_xy = y_true[..., 0:2]  # (sigma(t_xy) + c_xy)
        true_box_wh = y_true[..., 2:4]  # t_wh
        true_box_conf = tf.expand_dims(y_true[..., 4], 4)
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Compare each predicted box to all true boxes
        """
        # initially, drag all objectness of all boxes to 0
        conf_delta = pred_box_conf - 0

        # then, ignore the boxes which have good overlap with some true box
        true_xy = true_boxes[..., 0:2] / grid_factor
        true_wh = true_boxes[..., 2:4] / net_factor

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
        pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * self.anchors / net_factor, 4)

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
        conf_delta *= tf.expand_dims(tf.to_float(best_ious < self.ignore_thresh), 4)

        """
        Compute some online statistics
        """
        true_xy = true_box_xy / grid_factor
        true_wh = tf.exp(true_box_wh) * self.anchors / net_factor

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = pred_box_xy / grid_factor
        pred_wh = tf.exp(pred_box_wh) * self.anchors / net_factor

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
        iou_scores = object_mask * tf.expand_dims(iou_scores, 4)

        count = tf.reduce_sum(object_mask)
        count_noobj = tf.reduce_sum(1 - object_mask)
        detect_mask = tf.to_float((pred_box_conf * object_mask) >= 0.5)
        class_mask = tf.expand_dims(tf.to_float(tf.equal(tf.argmax(pred_box_class, -1), true_box_class)), 4)
        recall50 = tf.reduce_sum(tf.to_float(iou_scores >= 0.5) * detect_mask * class_mask) / (count + 1e-3)
        recall75 = tf.reduce_sum(tf.to_float(iou_scores >= 0.75) * detect_mask * class_mask) / (count + 1e-3)
        avg_iou = tf.reduce_sum(iou_scores) / (count + 1e-3)
        avg_obj = tf.reduce_sum(pred_box_conf * object_mask) / (count + 1e-3)
        avg_noobj = tf.reduce_sum(pred_box_conf * (1 - object_mask)) / (count_noobj + 1e-3)
        avg_cat = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3)

        """
        Warm-up training
        """
        batch_seen = tf.assign_add(batch_seen, 1.)

        true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches + 1),
                                                      lambda: [true_box_xy + (
                                                              0.5 + self.cell_grid[:, :grid_h, :grid_w, :, :]) * (
                                                                       1 - object_mask),
                                                               true_box_wh + tf.zeros_like(true_box_wh) * (
                                                                       1 - object_mask),
                                                               tf.ones_like(object_mask)],
                                                      lambda: [true_box_xy,
                                                               true_box_wh,
                                                               object_mask])

        """
        Compare each true box to all anchor boxes
        """
        wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1],
                                  axis=4)  # the smaller the box, the bigger the scale

        xy_delta = xywh_mask * (pred_box_xy - true_box_xy) * wh_scale * self.xywh_scale
        wh_delta = xywh_mask * (pred_box_wh - true_box_wh) * wh_scale * self.xywh_scale
        conf_delta = object_mask * (pred_box_conf - true_box_conf) * self.obj_scale + (
                1 - object_mask) * conf_delta * self.noobj_scale
        class_delta = object_mask * \
                      tf.expand_dims(
                          tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class),
                          4) * \
                      self.class_scale

        loss_xy = tf.reduce_sum(tf.square(xy_delta), list(range(1, 5)))
        loss_wh = tf.reduce_sum(tf.square(wh_delta), list(range(1, 5)))
        loss_conf = tf.reduce_sum(tf.square(conf_delta), list(range(1, 5)))
        loss_class = tf.reduce_sum(class_delta, list(range(1, 5)))

        loss = loss_xy + loss_wh + loss_conf + loss_class

        loss = tf.Print(loss, [grid_h, avg_obj], message='\navg_obj \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_noobj], message='avg_noobj \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_iou], message='avg_iou \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_cat], message='avg_cat \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, recall50], message='recall50 \t', summarize=1000)
        loss = tf.Print(loss, [grid_h, recall75], message='recall75 \t', summarize=1000)
        loss = tf.Print(loss, [grid_h, count], message='count \t', summarize=1000)
        loss = tf.Print(loss, [grid_h, tf.reduce_sum(loss_xy),
                               tf.reduce_sum(loss_wh),
                               tf.reduce_sum(loss_conf),
                               tf.reduce_sum(loss_class)], message='loss xy, wh, conf, class: \t', summarize=1000)

        return loss * self.grid_scale

    def compute_output_shape(self, input_shape):
        return [(None, 1)]


def create_scaled_mobilenet_model(
        nb_class,
        anchors,
        max_box_per_image,
        max_grid,
        batch_size,
        warmup_batches,
        ignore_thresh,
        grid_scales,
        obj_scale,
        noobj_scale,
        xywh_scale,
        class_scale
):
    input_image = Input(shape=(224, 224, 3))  # net_h, net_w, 3
    true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4))
    true_yolo_1 = Input(
        shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class))  # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_2 = Input(
        shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class))  # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_3 = Input(
        shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class))  # grid_h, grid_w, nb_anchor, 5+nb_class

    # layer 1 standard conv
    x = ZeroPadding2D(padding=(1, 1), name='conv1_zeropad')(input_image)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='valid', use_bias=False, strides=(2, 2),
               name='conv1')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation(relu6, name='conv1_relu6')(x)

    # layer 2-15
    # Conv dw / s1: filter shape (3 x 3 x 32 dw)    Conv pw / s1: filter shape (1 x 1 x 32 x 64)
    x = depthwise_separable_conv_block(x, dw_stride=(1, 1), pw_num_filter=64, id=2)
    # Conv dw / s2: filter shape (3 x 3 x 64 dw)    Conv pw / s1: filter shape (1 x 1 x 64 x 128)
    x = depthwise_separable_conv_block(x, dw_stride=(2, 2), pw_num_filter=128, id=4)
    # Conv dw / s1: filter shape (3 x 3 x 128 dw)   Conv pw / s1: filter shape (1 x 1 x 128 x 128)
    x = depthwise_separable_conv_block(x, dw_stride=(1, 1), pw_num_filter=128, id=6)
    # Conv dw / s2: filter shape (3 x 3 x 128 dw)   Conv pw / s1: filter shape (1 x 1 x 128 x 256)
    x = depthwise_separable_conv_block(x, dw_stride=(2, 2), pw_num_filter=256, id=8)
    # Conv dw / s1: filter shape (3 x 3 x 256 dw)   Conv pw / s1: filter shape (1 x 1 x 256 x 256)
    x = depthwise_separable_conv_block(x, dw_stride=(1, 1), pw_num_filter=256, id=10)

    # Conv dw / s2: filter shape (3 x 3 x 256 dw)   Conv pw / s1: filter shape (1 x 1 x 256 x 512)
    skip_28 = x
    x = depthwise_separable_conv_block(x, dw_stride=(2, 2), pw_num_filter=512, id=12)
    skip_14 = x
    x = depthwise_separable_conv_block(x, dw_stride=(2, 2), pw_num_filter=512, id=14)

    # layer 16-18  x: 14x14
    pred_yolo_1 = depthwise_conv_block(x, [
        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu6': True, 'layer_idx': 16},
        {'filter': (3 * (5 + nb_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'relu6': False, 'layer_idx': 17}])

    loss_yolo_1 = YoloLayer(anchors[12:],
                            [1 * num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            grid_scales[0],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_1, true_yolo_1, true_boxes])

    # layer 19-21
    x = depthwise_separable_conv_block(x, dw_stride=(1, 1), pw_num_filter=512, id=19)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_14])

    # layer 22-27
    # repeat 5 times
    x = depthwise_separable_conv_block(x, dw_stride=(1, 1), pw_num_filter=512, id=22)
    x = depthwise_separable_conv_block(x, dw_stride=(1, 1), pw_num_filter=512, id=24)
    x = depthwise_separable_conv_block(x, dw_stride=(1, 1), pw_num_filter=512, id=26)

    # layer 28-30
    pred_yolo_2 = depthwise_conv_block(x,
                              [{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu6': True, 'layer_idx': 28},
                               {'filter': (3 * (5 + nb_class)), 'kernel': 1, 'stride': 1, 'bnorm': False,
                                'relu6': False, 'layer_idx': 29}])
    loss_yolo_2 = YoloLayer(anchors[6:12],
                            [2 * num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            grid_scales[1],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_2, true_yolo_2, true_boxes])
    # layer 31-33
    x = depthwise_separable_conv_block(x, dw_stride=(1, 1), pw_num_filter=512, id=31)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_28])

    # layer 34-41
    x = depthwise_separable_conv_block(x, dw_stride=(1, 1), pw_num_filter=512, id=34)
    x = depthwise_separable_conv_block(x, dw_stride=(1, 1), pw_num_filter=512, id=36)

    # Conv dw / s2: filter shape (3 x 3 x 512 dw)   Conv pw / s1: filter shape (1 x 1 x 512 x 1024)
    x = depthwise_separable_conv_block(x, dw_stride=(1, 1), pw_num_filter=256, id=38)  # stride 2 -> 1

    # Conv dw / s2: filter shape (3 x 3 x 1024 dw)  Conv pw / s1: filter shape (1 x 1 x 1024 x 1024)
    x = depthwise_separable_conv_block(x, dw_stride=(1, 1), pw_num_filter=256, id=40)

    # layer 41-43
    pred_yolo_3 = depthwise_conv_block(x, [
        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu6': True, 'layer_idx': 38},
        {'filter': (3 * (5 + nb_class)), 'kernel': 1, 'stride': 1, 'bnorm': False,
         'relu6': False, 'layer_idx': 39}])
    loss_yolo_3 = YoloLayer(anchors[:6],
                            [4 * num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            grid_scales[2],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_3, true_yolo_3, true_boxes])

    train_model = Model([input_image, true_boxes, true_yolo_1, true_yolo_2, true_yolo_3],
                        [loss_yolo_1, loss_yolo_2, loss_yolo_3])
    infer_model = Model(input_image, [pred_yolo_1, pred_yolo_2, pred_yolo_3])

    return [train_model, infer_model]


def _conv_block(inp, convs, do_skip=True):
    x = inp
    count = 0

    for conv in convs:
        if count == (len(convs) - 2) and do_skip:
            skip_connection = x
        count += 1

        if conv['stride'] > 1:
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # unlike tensorflow darknet prefer left and top paddings
        x = Conv2D(conv['filter'],
                   conv['kernel'],
                   strides=conv['stride'],
                   padding='valid' if conv['stride'] > 1 else 'same',
                   # unlike tensorflow darknet prefer left and top paddings
                   name='conv_' + str(conv['layer_idx']),
                   use_bias=False if conv['bnorm'] else True)(x)
        if conv['bnorm']:
            x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
        if conv['relu6']:
            x = Activation(relu6, name='relu6_' + str(conv['layer_idx']))(x)

    return add([skip_connection, x]) if do_skip else x


def depthwise_conv_block(inputs, convs):
    x = inputs
    count = 0
    for conv in convs:
        count += 1
        # depthwise convolution layer
        if conv['stride'] > 1:  # only left and top padding
            x = ZeroPadding2D(padding=((1, 0), (1, 0)), name='yolo_dw_zeropad_%d' % conv['layer_idx'])(x)
        x = DepthwiseConv2D(kernel_size=conv['kernel'],
                            padding='valid' if conv['stride'] > 1 else 'same',
                            strides=conv['stride'],
                            use_bias=False, name='yolo_dw_conv_%d' % conv['layer_idx'])(x)
        if conv['bnorm']:
            x = BatchNormalization(epsilon=0.001, name='yolo_dw_bnorm_' + str(conv['layer_idx']))(x)
        if conv['relu6']:
            x = Activation(relu6, name='yolo_dw_relu6_' + str(conv['layer_idx']))(x)

        # pointwise convolution layer
        x = Conv2D(filters=conv['filter'],
                   padding='valid' if conv['stride'] > 1 else 'same',
                   kernel_size=(1, 1), strides=(1, 1), use_bias=False,
                   name='yolo_pw_conv_%d' % conv['layer_idx'])(x)
        x = BatchNormalization(axis=-1, name='yolo_pw_bn_%d' % conv['layer_idx'])(x)
        if conv['relu6']:  # alpha: negative slope coefficient
            x = Activation(relu6, name='yolo_pw_relu6_' + str(conv['layer_idx']))(x)

    return x


def depthwise_separable_conv_block(inputs, dw_stride, pw_num_filter, id, alpha=1.0, depth_multiplier=1):
    x = depthwise_block(inputs, dw_stride, id)
    x = pointwise_block(x, pw_num_filter, id)
    return x


def depthwise_block(inputs, stride, id, kernel=(3, 3), depth_multiplier=1):
    x = ZeroPadding2D(padding=(1, 1), name='convdw_zeropad_%d' % id)(inputs)
    x = DepthwiseConv2D(kernel_size=kernel, padding='valid', depth_multiplier=depth_multiplier, strides=stride,
                        use_bias=False, name='convdw_%d' % id)(x)
    x = BatchNormalization(axis=-1, name='convdw_bn_%d' % id)(x)
    return Activation(relu6, name='convdw_relu6_%d' % id)(x)


def pointwise_block(inputs, num_filter, id, kernel=(1, 1), strides=(1, 1), alpha=1.0):
    num_filter = int(alpha * num_filter)
    x = Conv2D(filters=num_filter, padding='same', kernel_size=kernel, strides=strides, use_bias=False,
               name='convpw_%d' % id)(inputs)
    x = BatchNormalization(axis=-1, name='convpw_bn_%d' % id)(x)
    return Activation(relu6, name='convpw_relu6_%d' % id)(x)


def relu6(x):
    return K.relu(x, max_value=6)


def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))