import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import caffe

caffe_root = os.environ["CAFFE_ROOT"]

MODEL_FILE = os.path.join(caffe_root,'examples/siamese/mnist_siamese.prototxt').encode()
# decrease if you want to preview during training
PRETRAINED_FILE = os.path.join(caffe_root,'examples/siamese/mnist_siamese_iter_50000.caffemodel').encode()

caffe.set_mode_cpu()
net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)

TEST_DATA_FILE = os.path.join(caffe_root,'data/mnist/t10k-images-idx3-ubyte')
TEST_LABEL_FILE = os.path.join(caffe_root,'data/mnist/t10k-labels-idx1-ubyte')

n = 10000

with open(TEST_DATA_FILE, 'rb') as f:
    f.read(16) # skip the header
    raw_data = np.fromstring(f.read(n * 28*28), dtype=np.uint8)

with open(TEST_LABEL_FILE, 'rb') as f:
    f.read(8) # skip the header
    labels = np.fromstring(f.read(n), dtype=np.uint8)

# reshape and preprocess
caffe_in = raw_data.reshape(n, 1, 28, 28) * 0.00390625 # manually scale data instead of using `caffe.io.Transformer`
out = net.forward_all(data=caffe_in)

feat = out['feat']
f = plt.figure(figsize=(16,9))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
     '#ff00ff', '#990000', '#999900', '#009900', '#009999']
for i in range(10):
    plt.plot(feat[labels==i,0].flatten(), feat[labels==i,1].flatten(), '.', c=c[i])
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.grid()
plt.show()