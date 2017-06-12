# -*- coding:utf-8 -*-
#
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)


from caffe2.python import core, workspace

import sys
sys.path.append('./models')
import squeezenet as mynet

init_net = mynet.init_net
predict_net = mynet.predict_net
print(type(init_net))
print(type(predict_net))

predict_net.name = "squeezenet_predict"
workspace.RunNetOnce(init_net)
workspace.CreateNet(predict_net)

init_net_str = init_net.SerializeToString()
predict_net_str = predict_net.SerializeToString()
print(type(init_net_str))
print(type(predict_net_str))
p = workspace.Predictor(unicode(init_net_str), unicode(predict_net_str))
print(type(p))
