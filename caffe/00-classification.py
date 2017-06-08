# -*- coding:utf-8 -*-
#
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import os

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)  # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

caffe_root = os.environ["CAFFE_ROOT"]

model_weights = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel').encode()
model_def = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/deploy.prototxt').encode()
assert (os.path.isfile(model_weights))
assert (os.path.isfile(model_def))

# print(model_def)
# print(model_weights)
# print(type(model_weights))
# print(type(caffe_root))
# print(sys.getdefaultencoding())
# Caffe require string input, unicode needs to be converted with encode()

caffe.set_mode_cpu()
print('Loading net')
print('        def:', model_def)
print('    weights:', model_weights)
net = caffe.Net(model_def,  # defines the structure of the model
                caffe.TEST,  # use test mode (e.g., don't perform dropout)
                weights=model_weights  # contains the trained weights
                )
print('Done!')

# load the mean ImageNet image (as distributed with Caffe) for subtraction
# shape=(3,256,256) input image resized
mu = np.load(os.path.join(caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
# shape=(3,)
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print('mean-subtracted values:', zip('BGR', mu))

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,  # batch size
                          3,  # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

image = caffe.io.load_image(os.path.join(caffe_root, 'examples/images/cat.jpg'))
transformed_image = transformer.preprocess('data', image)
# plt.imshow(image)
# plt.show()

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()
output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
class_id = output_prob.argmax()
print('predicted class is:', class_id)

# load ImageNet labels
labels_file = os.path.join(caffe_root, 'data/ilsvrc12/synset_words.txt')
assert (os.path.isfile(labels_file))
labels = np.loadtxt(labels_file, str, delimiter='\t')
print('output label:', labels[class_id])

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
print('probabilities and labels:')
print(zip(output_prob[top_inds], labels[top_inds]))

# ***  Examining intermediate output ***
# A net is not just a black box; let's take a look at some of the parameters and intermediate activations.
# First we'll see how to read out the structure of the net in terms of activation and parameter shapes.
# For each layer, let's look at the activation shapes, which typically have the form (batch_size, channel_dim, height, width).
# The activations are exposed as an OrderedDict, net.blobs.
# for each layer, show the output shape
print("output shape of each layer")
for layer_name, blob in net.blobs.iteritems():
    print(layer_name + '\t' + str(blob.data.shape))
# Now look at the parameter shapes. The parameters are exposed as another OrderedDict, net.params. We need to index the resulting values with either [0] for weights or [1] for biases.
# The param shapes typically have the form (output_channels, input_channels, filter_height, filter_width) (for the weights) and the 1-dimensional shape (output_channels,) (for the biases).
print()
print("weight shape and bias shape")
for layer_name, param in net.params.iteritems():
    print(layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape))
print("\nNet Layers Information")
from caffe import draw
for layer in net.layers:
    print(layer.type)
    if layer.type == 'Convolution':
        print(draw.get_layer_label(layer,'LR'))
    for blob in layer.blobs:
        print("   ", str(blob.data.shape))

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data);
    plt.axis('off')


# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))
plt.show()

feat = net.blobs['conv1'].data[0, :36]
vis_square(feat)
plt.show()

feat = net.blobs['pool5'].data[0]
vis_square(feat)
plt.show()

feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.title('fc6 dist')
plt.subplot(2, 1, 2)
plt.title('fc6 histogram')
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.show()

feat = net.blobs['prob'].data[0]
plt.figure(figsize=(15, 3))
plt.title("prob layer output")
plt.plot(feat.flat)
plt.show()