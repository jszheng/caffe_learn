# -*- coding:utf-8 -*-
#
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)


import numpy as np
import skimage.io
import skimage.transform
from matplotlib import pyplot
import os
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace

CAFFE2_SRC = "/home/manager/caffe2"
CAFFE2_ROOT = "/usr/local/caffe2"
CAFFE_MODELS = "./models"

MODEL = 'squeezenet', 'init_net.pb', 'predict_net.pb', 'ilsvrc_2012_mean.npy', 227
#MODEL = 'bvlc_alexnet', 'init_net.pb', 'predict_net.pb', 'ilsvrc_2012_mean.npy', 227
INPUT_IMAGE_SIZE = 227
IMAGE_LOCATION = '../images/flower-631765_1280.jpg'

# make sure all of the files are around...
INIT_NET = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[1])
print('INIT_NET = ', INIT_NET)
assert (os.path.isfile(INIT_NET))

PREDICT_NET = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[2])
print('PREDICT_NET = ', PREDICT_NET)
assert (os.path.isfile(PREDICT_NET))

IMAGE_NAME_TABLE = os.path.join(CAFFE_MODELS, 'alexnet_codes.txt')
print('Image Code File', IMAGE_NAME_TABLE)
assert (os.path.isfile(IMAGE_NAME_TABLE))


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def rescale(img, input_height, input_width):
    # scale short edge to the require size.
    # print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
    # print(("Model's input shape is %dx%d") % (input_height, input_width))
    aspect = img.shape[1] / float(img.shape[0])
    # print("Orginal aspect ratio: " + str(aspect))
    imgScaled = img
    if (aspect > 1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if (aspect < 1):
        # portrait orientation - tall image
        res = int(input_width / aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if (aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    pyplot.figure()
    pyplot.imshow(imgScaled)
    pyplot.axis('on')
    pyplot.title('Rescaled image')
    # print("New image shape:" + str(imgScaled.shape) + " in HWC")
    return imgScaled


# mean can be 128 or custom based on the model
# gives better results to remove the colors found in all of the training images
MEAN_FILE = os.path.join(CAFFE_MODELS, 'ilsvrc_2012_mean.npy')
if not os.path.exists(MEAN_FILE):
    mean = 128
else:
    mean = np.load(MEAN_FILE).mean(1).mean(1)
    mean = mean[:, np.newaxis, np.newaxis]
print("mean was set to: ", mean, type(mean))

# some models were trained with different image sizes, this helps you calibrate your image
# load and transform image
img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
print("Original Image", img.shape, img.dtype)
img = rescale(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
print("Rescaled Image", img.shape, img.dtype)
img = crop_center(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
print("After crop: ", img.shape, img.dtype)
pyplot.figure()
pyplot.imshow(img)
pyplot.axis('on')
pyplot.title('Cropped')

# switch to CHW
img = img.swapaxes(1, 2).swapaxes(0, 1)
pyplot.figure()
for i in range(3):
    # For some reason, pyplot subplot follows Matlab's indexing
    # convention (starting with 1). Well, we'll just follow it...
    pyplot.subplot(1, 3, i + 1)
    pyplot.imshow(img[i])
    pyplot.axis('off')
    pyplot.gray()
    pyplot.title('RGB channel %d' % (i + 1))

# switch to BGR
img = img[(2, 1, 0), :, :]

# remove mean for better results
img = img * 255 - mean

# add batch size
img = img[np.newaxis, :, :, :].astype(np.float32)
print("NCHW: ", img.shape, img.dtype)

# initialize the neural net
with open(INIT_NET, 'rb') as f:
    init_net = f.read()
with open(PREDICT_NET, 'rb') as f:
    predict_net = f.read()
print(type(init_net))
print(type(predict_net))
p = workspace.Predictor(init_net, predict_net)

# run the net and return prediction
results = p.run([img])

# turn it into something we can play with and examine which is in a multi-dimensional array
results = np.asarray(results)
print("results shape: ", results.shape)

# the rest of this is digging through the results
results = np.delete(results, 1)
index = 0
highest = 0
arr = np.empty((0, 2), dtype=object)
arr[:, 0] = int(10)
arr[:, 1:] = float(10)
for i, r in enumerate(results):
    # imagenet index begins with 1!
    i = i + 1
    arr = np.append(arr, np.array([[i, r]]), axis=0)
    if (r > highest):
        highest = r
        index = i
print(index, " :: ", highest)

# lookup the code and return the result
# top 3 results
# sorted(arr, key=lambda x: x[1], reverse=True)[:3]

# now we can grab the code list
with open(IMAGE_NAME_TABLE) as f:
    # and lookup our result from the list
    for line in f:
        code, result = line.partition(":")[::2]
        if (code.strip() == str(index)):
            print(result.strip()[1:-2])

pyplot.show()
