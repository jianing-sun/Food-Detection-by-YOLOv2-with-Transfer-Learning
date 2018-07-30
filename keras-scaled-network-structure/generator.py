import cv2
import numpy as np
import copy
from keras.utils import Sequence
from imgaug import augmenters as iaa
from utils.bbox import BoundBox, bbox_iou
from utils.image import apply_random_scale_and_crop, random_distort_image, random_flip, correct_bounding_boxes


class BatchGenerator(Sequence):
    def __init__(self,
                 instances,
                 anchors,   # for Feature Pyramid Networks we need 9 anchors, 3 for each scale
                 labels,
                 downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv1-3
                 max_box_per_image=30,
                 batch_size=1,
                 # min_net_size=224,
                 # max_net_size=224,
                 shuffle=True,
                 jitter=True,
                 norm=None
                 ):
        self.instances = instances
        self.batch_size = batch_size
        self.labels = labels
        self.downsample = downsample
        self.max_box_per_image = max_box_per_image
        # self.min_net_size = (min_net_size // self.downsample) * self.downsample
        # self.max_net_size = (max_net_size // self.downsample) * self.downsample
        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm
        self.anchors = [BoundBox(0, 0, anchors[2 * i], anchors[2 * i + 1]) for i in range(len(anchors) // 2)]
        self.net_h = 224
        self.net_w = 224

        # augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                sometimes(iaa.Affine(
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               # iaa.Invert(0.05, per_channel=True), # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               # change brightness of images (50-150% of original value)
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

        if shuffle:
            np.random.shuffle(self.instances)

    def __len__(self):
        return int(np.ceil(float(len(self.instances)) / self.batch_size))

    def __getitem__(self, idx):
        # get image input size, change every 10 batches (I didn't do this)
        # net_h, net_w = self._get_net_size(idx)
        net_h, net_w = self.net_h, self.net_w
        base_grid_h, base_grid_w = net_h // self.downsample, net_w // self.downsample

        # determine the first and the last indices of the batch
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        x_batch = np.zeros((r_bound - l_bound, net_h, net_w, 3))  # input images
        t_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.max_box_per_image, 4))  # list of groundtruth boxes

        # initialize the inputs and the outputs
        yolo_1 = np.zeros((r_bound - l_bound, 1 * base_grid_h, 1 * base_grid_w, len(self.anchors) // 3,
                           4 + 1 + len(self.labels)))  # desired network output 1
        yolo_2 = np.zeros((r_bound - l_bound, 2 * base_grid_h, 2 * base_grid_w, len(self.anchors) // 3,
                           4 + 1 + len(self.labels)))  # desired network output 2
        yolo_3 = np.zeros((r_bound - l_bound, 4 * base_grid_h, 4 * base_grid_w, len(self.anchors) // 3,
                           4 + 1 + len(self.labels)))  # desired network output 3
        yolos = [yolo_3, yolo_2, yolo_1]

        dummy_yolo_1 = np.zeros((r_bound - l_bound, 1))
        dummy_yolo_2 = np.zeros((r_bound - l_bound, 1))
        dummy_yolo_3 = np.zeros((r_bound - l_bound, 1))

        instance_count = 0
        true_box_index = 0

        # do the logic to fill in input and output
        for train_instance in self.instances[l_bound:r_bound]:
            # augment input image and fix object's position and size
            # img, all_objs = self._aug_image(train_instance, net_h, net_w)
            img, all_objs = self.aug_image(train_instance, jitter=self.jitter)

            for obj in all_objs:
                # find the best anchor box for this object
                max_anchor = None
                max_index = -1
                max_iou = -1

                shifted_box = BoundBox(0,
                                       0,
                                       obj['xmax'] - obj['xmin'],
                                       obj['ymax'] - obj['ymin'])

                for i in range(len(self.anchors)):
                    anchor = self.anchors[i]
                    iou = bbox_iou(shifted_box, anchor)

                    if max_iou < iou:
                        max_anchor = anchor
                        max_index = i
                        max_iou = iou

                        # determine the yolo to be responsible for this bounding box
                yolo = yolos[max_index // 3]
                grid_h, grid_w = yolo.shape[1:3]

                # determine the position of the bounding box on the grid
                center_x = .5 * (obj['xmin'] + obj['xmax'])
                center_x = center_x / float(net_w) * grid_w  # sigma(t_x) + c_x
                center_y = .5 * (obj['ymin'] + obj['ymax'])
                center_y = center_y / float(net_h) * grid_h  # sigma(t_y) + c_y

                # determine the sizes of the bounding box
                w = np.log((obj['xmax'] - obj['xmin']) / float(max_anchor.xmax))  # t_w
                h = np.log((obj['ymax'] - obj['ymin']) / float(max_anchor.ymax))  # t_h

                box = [center_x, center_y, w, h]

                # determine the index of the label
                obj_indx = self.labels.index(obj['name'])

                # determine the location of the cell responsible for this object
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                # assign ground truth x, y, w, h, confidence and class probs to y_batch
                yolo[instance_count, grid_y, grid_x, max_index % 3] = 0
                yolo[instance_count, grid_y, grid_x, max_index % 3, 0:4] = box
                yolo[instance_count, grid_y, grid_x, max_index % 3, 4] = 1.
                yolo[instance_count, grid_y, grid_x, max_index % 3, 5 + obj_indx] = 1

                # assign the true box to t_batch
                true_box = [center_x, center_y, obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin']]
                t_batch[instance_count, 0, 0, 0, true_box_index] = true_box

                true_box_index += 1
                true_box_index = true_box_index % self.max_box_per_image

            # assign input image to x_batch
            if self.norm is not None:
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    font = cv2.FONT_HERSHEY_SIMPLEX  # use default font
                    # cv2.rectangle(img, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (255, 0, 0), 1)
                    cv2.rectangle(img[:, :, ::-1], (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']),
                                  (255, 0, 0), 2)
                    # cv2.rectangle(img, (obj['xmin'] + 2, obj['ymax'] - 12), (obj['xmax'], obj['ymax']), (0, 0, 255), thickness=-1)
                    cv2.putText(img[:, :, ::-1], obj['name'],
                                (obj['xmin'] + 2, obj['ymin'] + 12),
                                font, 1.2e-3 * img.shape[0],
                                (255, 255, 255), 1)

                x_batch[instance_count] = img

            # increase instance counter in the current batch
            instance_count += 1

        return [x_batch, t_batch, yolo_1, yolo_2, yolo_3], [dummy_yolo_1, dummy_yolo_2, dummy_yolo_3]

    def _get_net_size(self, idx):
        if idx % 10 == 0:
            net_size = self.downsample * np.random.randint(self.min_net_size / self.downsample, \
                                                           self.max_net_size / self.downsample + 1)
            print("resizing: ", net_size, net_size)
            self.net_h, self.net_w = net_size, net_size
        return self.net_h, self.net_w

    def _aug_image(self, instance, net_h, net_w):
        image_name = instance['filename']
        image = cv2.imread(image_name)  # RGB image

        if image is None:
            print('Cannot find ', image_name)
        image = image[:, :, ::-1]  # RGB image

        image_h, image_w, _ = image.shape

        # determine the amount of scaling and cropping
        dw = self.jitter * image_w
        dh = self.jitter * image_h

        new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh))
        scale = np.random.uniform(0.25, 2)

        if new_ar < 1:
            new_h = int(scale * net_h)
            new_w = int(net_h * new_ar)
        else:
            new_w = int(scale * net_w)
            new_h = int(net_w / new_ar)

        dx = int(np.random.uniform(0, net_w - new_w))
        dy = int(np.random.uniform(0, net_h - new_h))

        # apply scaling and cropping
        im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)

        # randomly distort hsv space
        im_sized = random_distort_image(im_sized)

        # randomly flip
        flip = np.random.randint(2)
        im_sized = random_flip(im_sized, flip)

        # correct the size and pos of bounding boxes
        all_objs = correct_bounding_boxes(instance['object'], new_w, new_h, net_w, net_h, dx, dy, flip, image_w,
                                          image_h)

        return im_sized, all_objs

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        image = cv2.imread(image_name)

        if image is None:
            print('Cannot find ', image_name)

        h, w, c = image.shape
        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:   # by default jitter=0.1
            # scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

            # translate the image
            max_offx = (scale - 1.) * w
            max_offy = (scale - 1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = image[offy: (offy + h), offx: (offx + w)]

            # flip the image with 50% chance
            flip = np.random.binomial(1, .5)
            if flip > 0.5:
                image = cv2.flip(image, 1)

            # image = self.aug_pipe.augment_image(image)

        # resize the image to standard size
        image = cv2.resize(image, (self.net_h, self.net_w))
        image = image[:, :, ::-1]

        # fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if jitter:
                    obj[attr] = int(obj[attr] * scale - offx)

                obj[attr] = int(obj[attr] * float(self.net_w) / w)
                obj[attr] = max(min(obj[attr], self.net_w), 0)

            for attr in ['ymin', 'ymax']:
                if jitter:
                    obj[attr] = int(obj[attr] * scale - offy)

                obj[attr] = int(obj[attr] * float(self.net_h) / h)
                obj[attr] = max(min(obj[attr], self.net_h), 0)

            if jitter and flip > 0.5:
                xmin = obj['xmin']
                obj['xmin'] = self.net_w - obj['xmax']
                obj['xmax'] = self.net_w - xmin

        return image, all_objs

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.instances)

    def num_classes(self):
        return len(self.labels)

    def size(self):
        return len(self.instances)

    def get_anchors(self):
        anchors = []

        for anchor in self.anchors:
            anchors += [anchor.xmax, anchor.ymax]

        return anchors

    def load_annotation(self, i):
        annots = []

        for obj in self.instances[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.labels.index(obj['name'])]
            annots += [annot]

        if len(annots) == 0:
            annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.instances[i]['filename'])