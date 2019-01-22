import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import inception_v3
from keras.models import Model, load_model
from matplotlib.patches import Rectangle
from scipy import ndimage
from skimage import io, transform

from cpu_nms import cpu_nms

# INPUT_FILENAME, OUTPUT_FILENAME = sys.argv[1], sys.argv[2]
input_path = './uecfood100/4.jpg'  # './input_img/bread.jpg'
output_path = './uecfood100_out/4_out.jpg'


def plot_bbox_heatmap(im, bbox_list, im_result, height=10, width=4):
    f = plt.figure(1)
    f.set_size_inches(height, width)

    # im = misc.imread(img_path)
    ax = plt.subplot(1, 2, 1)
    plt.imshow(im)

    # remove axis
    plt.gca().xaxis.set_major_locator(plt.NullLocator)
    plt.gca().yaxis.set_major_locator(plt.NullLocator)

    ax_fam = plt.subplot(1, 2, 2)
    plt.imshow(im_result)

    # remove axis
    plt.gca().xaxis.set_major_locator(plt.NullLocator)
    plt.gca().yaxis.set_major_locator(plt.NullLocator)

    for box in bbox_list:
        ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                               facecolor="none", edgecolor='blue', linewidth=3.0))
    plt.savefig('xxx.png')
    return f


def bbox(img, mode='width_height'):
    '''
        Returns a bounding box covering all the non-zero area in the image.
        "mode" : "width_height" returns width in [2] and height in [3], "max" returns xmax in [2] and ymax in [3]
    '''
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y, ymax = np.where(rows)[0][[0, -1]]
    x, xmax = np.where(cols)[0][[0, -1]]

    if mode == 'width_height':
        return x, y, xmax - x, ymax - y
    elif mode == 'max':
        return x, y, xmax, ymax


def generateBBoxfromCAM(cams, reshape_size=(224, 224), percentage_heat=0.4, size_restriction=0.1,
                        box_expansion=0.1):
    predicted_bboxes = []
    predicted_scores = []

    # heatmap = resize(heatmap, tuple(reshape_size), order=1, preserve_range=True)
    bb_thres = np.max(cams) * percentage_heat

    binary_heat = cams
    binary_heat = np.where(binary_heat > bb_thres, 255, 0)

    min_size = reshape_size[0] * reshape_size[1] * size_restriction
    labeled, nr_objects = ndimage.label(binary_heat)
    [objects, counts] = np.unique(labeled, return_counts=True)
    biggest_components = np.argsort(counts[1:])[::-1]
    selected_components = [1 if counts[i + 1] >= min_size else 0 for i in
                           biggest_components]
    biggest_components = biggest_components[:min([np.sum(selected_components), 9999])]
    # cams = cams / 255.0

    # Get bboxes
    for selected, comp in zip(selected_components, biggest_components):
        if selected:
            max_heat = np.where(labeled == comp + 1, 255, 0)  # get the biggest

            box = list(bbox(max_heat))

            # expand box before final detection
            x_exp = box[2] * box_expansion
            y_exp = box[3] * box_expansion
            box[0] = int(max([0, box[0] - x_exp / 2]))
            box[1] = int(max([0, box[1] - y_exp / 2]))
            # change width and height by xmax and ymax
            box[2] += box[0]
            box[3] += box[1]
            box[2] = int(min([reshape_size[1] - 1, box[2] + x_exp]))
            box[3] = int(min([reshape_size[0] - 1, box[3] + y_exp]))

            predicted_bboxes.append(box)

            # Get score for current bbox
            score = np.mean(cams[box[1]:box[3], box[0]:box[2]])  # use mean CAM value of the bbox as a score
            predicted_scores.append(score)

    # Now apply NMS on all the obtained bboxes
    nms_threshold = 0.3
    # logging.info('bboxes before NMS: '+str(len(predicted_scores)))
    if len(predicted_scores) > 0:
        dets = np.hstack((np.array(predicted_bboxes), np.array(predicted_scores)[:, np.newaxis])).astype(np.float32)

        keep = cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        predicted_bboxes = []
        predicted_scores = []
        for idet in range(dets.shape[0]):
            predicted_bboxes.append(dets[idet, :4])
            predicted_scores.append(dets[idet, -1])
            # logging.info('bboxes after NMS: '+str(len(predicted_scores)))

    return [predicted_bboxes, predicted_scores]


# Preprocessing functions for images
def preprocess_im(im, input_size=(499, 499, 3)):  # 299x299 = 5x5 resolution; 349x349 = 7x7 resolution  499x499=14x14
    im = transform.resize(im, output_shape=input_size)
    im = im * 255.0
    im = inception_v3.preprocess_input(im)  # tf mode: scale image from -1 to 1
    return np.expand_dims(im, axis=0)


# Rescale array to 0-1
def rescale_arr(arr):
    min_, max_ = np.min(arr), np.max(arr)
    arr = arr - min_
    arr = arr / (max_ - min_)
    return arr


''' code start from here '''

# Load the model
# model = load_model('./notebooks/models/model_foodvsnot_v3.h5')
# model = load_model('./model_food.h5')
model = load_model('./notebooks/models/my_gap_foodnotfood.h5')
print(model.summary())

# Get the model to get the maps
model_maps = Model(inputs=model.input, outputs=model.layers[-3].input)  # TODO
print('')
print('model_maps')
print(model_maps.summary())

# Load the input image
# im = io.imread(input_path)
# im = cv2.resize(cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB), (224, 224))
# im = cv2.resize(cv2.imread(input_path), (224, 224))
# im = cv2.resize(cv2.imread(input_path), (672, 672))
im = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
im = cv2.resize(cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB), (499, 499))

# Get the maps
maps = model_maps.predict(preprocess_im(im))
maps = np.squeeze(maps)

# Get the weights (to scale each map)
layer_weights = model.layers[-1].get_weights()[0]  # Weights of the FC layer
scaling_weights = layer_weights[:, 1]

# Multiply
scaled_maps = maps * scaling_weights  # maps (8, 8, 2048)  scaling_weights (2048, )

# Take mean from (5, 5, 2048) -> (5, 5)
cam_map = np.mean(scaled_maps, axis=2)

# Resize it back to original image size
cam_map = rescale_arr(cam_map)   # rescale array to 0-1
# cam_map = transform.resize(cam_map, output_shape=(224, 224))
cam_map = transform.resize(cam_map, output_shape=im.shape[:-1])

# extract bounding boxes from cam
[predicted_bboxes, predicted_scores] = generateBBoxfromCAM(cam_map, reshape_size=im.shape[:-1])

# Show it superimposed on heatmap
heatmap = cv2.applyColorMap((cam_map * 255.0).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]

im_result = (im / 255.0) * 0.5 + (heatmap / 255.0) * 0.5
# im_result = heatmap / 255.0


# Prediction score
p = model.predict(preprocess_im(im))[0, 1]
print('Food Probability Score:', p)

# Save the image
io.imsave(output_path, im_result)
plot_bbox_heatmap(im, predicted_bboxes, im_result)



