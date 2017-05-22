# -*- coding: utf-8 -*-

import caffe
import numpy as np

#deploy      = 'lenet_train_test.prototxt'  # deploy文件
deploy      = 'deploy.prototxt'  # deploy文件
caffe_model = 'lenet_iter_10000.caffemodel'  # 训练好的 caffemodel

net = caffe.Net(deploy, caffe.TEST, weights=caffe_model)  # 加载model和network

for k, v in net.params.items():  # 查看各层参数规模
    print(k, v[0].data.shape)

w1 = net.params['Convolution1'][0].data  # 提取参数w
b1 = net.params['Convolution1'][1].data  # 提取参数b
net.forward()  # 运行测试

for k, v in net.blobs.items():  # 查看各层数据规模
    print(k, v.data.shape)

fea = net.blobs['InnerProduct1'].data  # 提取某层数据（特征）
print(fea)