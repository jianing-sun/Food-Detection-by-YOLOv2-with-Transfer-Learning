"""
Created by Jianing Sun, May 13th
"""

import os
import PIL.Image
import numpy as np


def save_classes(path):
    with open(path) as f:
        result = []
        class_names = f.readlines()
        for class_name in class_names:
            result.append(class_name.split()[0])
    result = np.array(result)
    np.savetxt("unimin2016_classes.txt", result, fmt="%s")


def create_label_dict(class_path):
    label_dict = {}
    with open(class_path) as f:
        class_names = f.readlines()
    for i in range(0, len(class_names)):
        label_dict[class_names[i][:-1]] = i
    return label_dict


def coordinates_convertion(four_coordinates_str):
    coordinates = four_coordinates_str.split(' ')
    coordinates[:] = (value for value in coordinates if value != '')
    coordinates = [int(x) for x in coordinates]
    xs = coordinates[::2]
    ys = coordinates[1::2]
    xmin = np.min(xs)
    xmax = np.max(xs)
    ymin = np.min(ys)
    ymax = np.max(ys)
    out = [xmin, ymin, xmax, ymax]
    out = ' '.join(str(x) for x in out)
    return out


def csv2data(label_dict):
    with open('./useful_data_info.csv') as f:
        entries = f.readlines()
        out = {}
        for entry in entries:
            entry = entry[:-1].split(',')
            entry[2] = coordinates_convertion(entry[2])
            entry[1] = entry[1] + ' ' + entry[2]
            entry = entry[:2]
            if entry[0] in out.keys():
                out[entry[0]].append(entry[1])
            else:
                out[entry[0]] = [entry[1]]

    image_data = list()
    index = 0
    for k, v in out.items():
        image_data.append([k])
        for i in v:
            image_data[index].append(i)
        index += 1

    for no, entry in enumerate(image_data):
        for i, box in enumerate(entry):
            if i != 0:
                # print(box)
                box = box.split(' ')
                box[:] = (value for value in box if value != '')
                box[0] = label_dict[box[0]]  # convert class name to numbers (0~)

                for k in range(1, 5):  # Change box boundaries from str to int
                    box[k] = int(box[k])

                image_data[no][i] = box
    return image_data


def load_images(image_data):
    images = []
    for i, data in enumerate(image_data):
        img = PIL.Image.open(os.path.join('/Users/jianingsun/Documents/Research/May/'
                                          'Dataset/UNIMIB2016/original/', data[0] + '.jpg'))
        img = np.array(img, dtype=np.uint8)
        images.append(img)


def images2npv(images, image_data, shuffle=False):
    images = np.array(images, dtype=np.uint8)
    image_data = [np.array(image_data[i][1:]) for i in range(images.shape[0])]
    image_data = np.array(image_data)
    # shuffle dataset
    if shuffle:
        np.random.seed(13)
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        images, image_data = images[indices], image_data[indices]
    print('dataset contains {} images'.format(images.shape[0]))
    np.savez('unimib2016-js', image=images, boxes=image_data)
    print('npz file has been generated and saved as unimib2016-js.npz')


if __name__ == '__main__':
    save_classes('./statistics_cell.txt')
    label_dict = create_label_dict('./unimin2016_classes.txt')
    # print(label_dict)
    image_data = csv2data(label_dict)
    load_images(image_data)
