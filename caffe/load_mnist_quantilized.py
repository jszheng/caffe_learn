# -*- coding:utf-8 -*-
#
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pylab import *
import struct
import caffe

def binary(num):
    # Struct can provide us with the float packed into bytes. The '!' ensures that
    # it's in network byte order (big-endian) and the 'f' says that it should be
    # packed as a float. Alternatively, for double-precision, you could use 'd'.
    packed = struct.pack('!f', num)
    print('Packed: %s' % repr(packed))

    # For each character in the returned string, we'll turn it into its corresponding
    # integer code point
    #
    # [62, 163, 215, 10] = [ord(c) for c in '>\xa3\xd7\n']
    integers = [ord(c) for c in packed]
    print('Integers: %s' % integers)

    # For each integer, we'll convert it to its binary representation.
    binaries = [bin(i) for i in integers]
    print('Binaries: %s' % binaries)

    # Now strip off the '0b' from each of these
    stripped_binaries = [s.replace('0b', '') for s in binaries]
    print('Stripped: %s' % stripped_binaries)

    # Pad each byte's binary representation's with 0's to make sure it has all 8 bits:
    #
    # ['00111110', '10100011', '11010111', '00001010']
    padded = [s.rjust(8, '0') for s in stripped_binaries]
    print('Padded: %s' % padded)

    # At this point, we have each of the bytes for the network byte ordered float
    # in an array as binary strings. Now we just concatenate them to get the total
    # representation of the float:
    return ''.join(padded)

model_def = str('mnist/quantized.prototxt')
model_weights = str('mnist/lenet_fine_tune_iter_10000.caffemodel')
assert (os.path.isfile(model_weights))
assert (os.path.isfile(model_def))

caffe.set_mode_cpu()
print('Loading net')
print('        def:', model_def)
print('    weights:', model_weights)

net = caffe.Net(model_def, model_weights, caffe.TEST)

print('Done!')

print("output shape of each layer")
for layer_name, blob in net.blobs.iteritems():
    print(layer_name + '\t' + str(blob.data.shape))
print()
print("weight shape and bias shape")
for layer_name, param in net.params.iteritems():
    print(layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape))

print("\nlook inside")
for layer_name, param in net.params.iteritems():
    print(layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape))
    #print(len(param))      # 2 BlobVec
    #print(type(param[0]))  # Blob
    coeff = param[0].data
    bias = param[1].data
    print(coeff.shape)
    c = coeff.flat[0]
    print(c)
    print(binary(c)) # not quantilized, it is done in forward path, user should do that.
    print(bias[0])
    print()

print(type(net.layers))
for layer in net.layers:
    print(layer.type)
    for blob in layer.blobs:
        print("   ", str(blob.data.shape))