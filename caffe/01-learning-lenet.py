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
from pylab import *
import time

caffe_root = os.environ["CAFFE_ROOT"]

from caffe import layers as L, params as P


def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1. / 255), ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()


with open('mnist/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_train_lmdb', 64)))

with open('mnist/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_test_lmdb', 100)))

#caffe.set_device(0)
#caffe.set_mode_cpu()

### load the solver and create train and test nets
solver = caffe.SGDSolver(str('01_lenet_solver.prototxt'))

# each output is (batch size, feature dim, spatial dim)
print("solver net blobs shape")
for k, v in solver.net.blobs.items():
    print(k, v.data.shape)


# just print the weight sizes (we'll omit the biases)
print("solver net pararmter shape")
for k, v in solver.net.params.items():
    print(k, v[0].data.shape)

solver.net.forward()  # train net
fwd_ret = solver.test_nets[0].forward()  # test net (there can be more than one)
print(fwd_ret)

# # check training and test data set : image and label are all ok
# imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
# print('train labels:', solver.net.blobs['label'].data[:8])
# show()
#
# imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
# print('test labels:', solver.test_nets[0].blobs['label'].data[:8])
# show()
#
# # try one step
# solver.step(1)
# imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5)
#        .transpose(0, 2, 1, 3).reshape(4*5, 5*5), cmap='gray'); axis('off')
# show()

# timer
start_time = time.time()

# training
niter = 200
test_interval = 25
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 8, 10))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data

    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['score'].data[:8]

    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4
        print('Iteration', it, 'test accuracy', correct / 1e4)

elapsed_time = time.time() - start_time
print('Elapsed Time = ', elapsed_time)

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
show()

# not clear
# for i in range(8):
#     figure(figsize=(2, 2))
#     imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
#     figure(figsize=(10, 2))
#     imshow(output[:50, i].T, interpolation='nearest', cmap='gray')
#     xlabel('iteration')
#     ylabel('label')
# show()

# darker
for i in range(8):
    figure(figsize=(2, 2))
    imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    figure(figsize=(10, 2))
    imshow(exp(output[:50, i].T) / exp(output[:50, i].T).sum(0), interpolation='nearest', cmap='gray')
    xlabel('iteration')
    ylabel('label')