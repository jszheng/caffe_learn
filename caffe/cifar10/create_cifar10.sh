#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.
set -e

DBTYPE=lmdb

echo "Creating $DBTYPE..."
rm -rf cifar10_test_$DBTYPE
rm -rf cifar10_train_$DBTYPE
$CAFFE_ROOT/build/examples/cifar10/convert_cifar_data.bin $CAFFE_ROOT/data/cifar10 . lmdb

echo "Computing image mean..."
$CAFFE_ROOT/build/tools/compute_image_mean -backend=$DBTYPE cifar10_train_$DBTYPE mean.binaryproto

echo "Done."
