import numpy as np
import matplotlib.pyplot as plt
import caffe
import os
import sys

im = caffe.io.load_image('images/cat.jpg')
print im.shape
# (360, 480, 3)
plt.imshow(im)
plt.axis('off')
plt.show()

net = caffe.Net('net_surgery/conv.prototxt', caffe.TEST)
# add axis as (1, 360, 480, 3)
# transpose axis order 0->1, 3->3, 1->360, 2->480
# now becomes (1, 3, 360, 480)
im_input = im[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
print "data-blobs:", im_input.shape

net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input

# change (3, 360, 480) to (360, 480, 3) for display
plt.imshow(net.blobs['data'].data[0].transpose(1, 2, 0))
plt.axis('off')
plt.show()

plt.rcParams['image.cmap'] = 'gray'


def show_data(data, head, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.figure()
    plt.title(head)
    plt.imshow(data)
    plt.axis('off')


print "data-blobs:", net.blobs['data'].data.shape
show_data(net.blobs['data'].data[0], 'origin images')
plt.show()

net.forward()
print "data-blobs:", net.blobs['data'].data.shape
print "conv-blobs:", net.blobs['conv'].data.shape
print "weight-blobs:", net.params['conv'][0].data.shape
show_data(net.params['conv'][0].data[:, 0], 'conv weights(filter)')
show_data(net.blobs['conv'].data[0], 'post-conv images')
plt.show()