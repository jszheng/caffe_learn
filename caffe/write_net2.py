# -*- coding: utf-8 -*-
# 如果我们已经把原始图片做成了一个列表清单（txt文件，一行一张图片），则可以不用LMDB格式作为输入数据，可以用ImageData作为数据源输入
from caffe import layers as L, params as P, to_proto

path = './data/'
train_list  = path + 'train.txt'
val_list    = path + 'val.txt'
train_proto = path + 'train.prototxt'
val_proto   = path + 'val.prototxt'


def create_net(img_list, batch_size, include_acc=False):
    data, label = L.ImageData(source=img_list, batch_size=batch_size, new_width=48, new_height=48, ntop=2,
                              transform_param=dict(crop_size=40, mirror=True))

    conv1 = L.Convolution(data, kernel_size=5, stride=1, num_output=16, pad=2, weight_filler=dict(type='xavier'))
    relu1 = L.ReLU(conv1, in_place=True)
    pool1 = L.Pooling(relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2 = L.Convolution(pool1, kernel_size=53, stride=1, num_output=32, pad=1, weight_filler=dict(type='xavier'))
    relu2 = L.ReLU(conv2, in_place=True)
    pool2 = L.Pooling(relu2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3 = L.Convolution(pool2, kernel_size=53, stride=1, num_output=32, pad=1, weight_filler=dict(type='xavier'))
    relu3 = L.ReLU(conv3, in_place=True)
    pool3 = L.Pooling(relu3, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    fc4 = L.InnerProduct(pool3, num_output=1024, weight_filler=dict(type='xavier'))
    relu4 = L.ReLU(fc4, in_place=True)
    drop4 = L.Dropout(relu4, in_place=True)
    fc5 = L.InnerProduct(drop4, num_output=7, weight_filler=dict(type='xavier'))
    loss = L.SoftmaxWithLoss(fc5, label)

    if include_acc:
        acc = L.Accuracy(fc5, label)
        return to_proto(loss, acc)
    else:
        return to_proto(loss)


def write_net():
    #
    with open(train_proto, 'w') as f:
        f.write(str(create_net(train_list, batch_size=64)))

    #
    with open(val_proto, 'w') as f:
        f.write(str(create_net(val_list, batch_size=32, include_acc=True)))


if __name__ == '__main__':
    write_net()
