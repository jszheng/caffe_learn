#
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import os
# You should have checked out original Caffe
# git clone https://github.com/BVLC/caffe.git
# change the CAFFE_ROOT directory below accordingly
CAFFE_ROOT = os.path.expanduser('/home/manager/caffe')

if not os.path.exists(CAFFE_ROOT):
    print("Houston, you may have a problem.")
    print("Did you change CAFFE_ROOT to point to your local Caffe repo?")
    print("Try running: git clone https://github.com/BVLC/caffe.git")

# Pick a model, and if you don't have it, it will be downloaded
# format below is the model's folder, model's dataset inside that folder
# MODEL = 'bvlc_alexnet', 'bvlc_alexnet.caffemodel'
# MODEL = 'bvlc_googlenet', 'bvlc_googlenet.caffemodel'
# MODEL = 'finetune_flickr_style', 'finetune_flickr_style.caffemodel'
# MODEL = 'bvlc_reference_caffenet', 'bvlc_reference_caffenet.caffemodel'
MODEL = 'bvlc_reference_rcnn_ilsvrc13', 'bvlc_reference_rcnn_ilsvrc13.caffemodel'

# scripts to download the models reside here (~/caffe/models)
# after downloading the data will exist with the script
CAFFE_MODELS = os.path.join(CAFFE_ROOT, 'models')

# this is like: ~/caffe/models/bvlc_alexnet/deploy.prototxt
CAFFE_MODEL_FILE = os.path.join(CAFFE_MODELS, MODEL[0], 'deploy.prototxt')
# this is like: ~/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel
CAFFE_PRETRAINED = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[1])

# if the model folder doesn't have the goods, then download it
# this is usually a pretty big file with the .caffemodel extension
if not os.path.exists(CAFFE_PRETRAINED):
    print(CAFFE_PRETRAINED + " not found. Attempting download. Be patient...")
    print(
        os.path.join(CAFFE_ROOT, 'scripts/download_model_binary.py') +
        ' ' +
        os.path.join(CAFFE_ROOT, 'models', MODEL[0]))
else:
    print("You already have " + CAFFE_PRETRAINED)

# if the .prototxt file was missing then you're in trouble; cannot continue
if not os.path.exists(CAFFE_MODEL_FILE):
    print("Caffe model file, " + CAFFE_MODEL_FILE + " was not found!")
else:
    print("Now we can test the model!")