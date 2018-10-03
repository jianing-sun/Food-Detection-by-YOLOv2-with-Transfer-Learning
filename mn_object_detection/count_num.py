import os
# count the number of objects from .xml annotation
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.utils import Sequence


def parse_xml_num_object(ann_dir, img_dir):
    all_imgs = []
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}
        tree = ET.parse(ann_dir + ann)
        num = 0
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag:
                num += 1
        img['obj_num'] = num
        if num > 0:
            all_imgs += [img]
    return all_imgs


class CounterBatchGenerator(Sequence):
    def __init__(self, images,
                 config,
                 shuffle=True,
                 jitter=True,
                 norm=None):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm

        if shuffle:
            np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images)) / self.config['BATCH_SIZE']))

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images)

    def load_image(self, i):
        return cv2.imread(self.images[i]['filename'])

    def __getitem__(self, idx):
        l_bound = idx * self.config['BATCH_SIZE']
        r_bound = (idx + 1) * self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))  # input images
        y_batch = np.zeros((r_bound - l_bound, 1))  # desired network output

        for train_instance in self.images[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, obj_num = self.aug_image(train_instance, jitter=self.jitter)

            # assign ground truth obj_num to y_batch
            y_batch[instance_count, 0] = obj_num

            # assign input image to x_batch
            if self.norm != None:
                x_batch[instance_count] = self.norm(img)
            else:
                # plot number of objects on image to check if the code is right
                cv2.putText(img[:, :, ::-1], str(obj_num), (100, 100), 0, 1, (255, 0, 0), 2)
                x_batch[instance_count] = img

            # increase instance counter in current batch
            instance_count += 1

        return [x_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        image = cv2.imread(image_name)

        if image is None:
            print('Warning: Cannot find ', image_name)

        # resize the image to standard size
        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
        image = image[:, :, ::-1]

        return image, train_instance['obj_num']


# generate CounterBatchGenerator for training and validation set
def read_category():
    category = []
    with open('/Volumes/JS/UECFOOD100_JS/category.txt', 'r') as file:
        for i, line in enumerate(file):
            if i > 0:
                line = line.rstrip('\n')
                line = line.split('\t')
                category.append(line[1])
    return category


LABELS = read_category()

generator_config = {
    'IMAGE_H': 224,
    'IMAGE_W': 224,
    'LABELS': LABELS,
    'CLASS': len(LABELS),
    'BATCH_SIZE': 16,
}

# generate all_imgs (array), each instance is a dictionary
all_imgs = []
for i in range(0, 100):
    image_path = '/Volumes/JS/UECFOOD100_JS/' + str(i + 1) + '/'
    annot_path = '/Volumes/JS/UECFOOD100_JS/' + str(i + 1) + '/' + '/annotations_new/'

    folder_imgs = parse_xml_num_object(annot_path, image_path)
    all_imgs.extend(folder_imgs)
print(np.array(all_imgs).shape)

batches = CounterBatchGenerator(all_imgs, generator_config, jitter=False)
img = batches[0][0][0][1]
plt.imshow(img.astype('uint8'))
plt.show()
