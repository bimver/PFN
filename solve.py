import caffe
import surgery

import numpy as np
import os
import sys



weights = './VGG_ILSVRC_16_layers.caffemodel'
proto = './PFN.prototxt'
#weights = '/media/fangzheng/fang/code/fcn.berkeleyvision.org-master/ilsvrc-nets/vgg16-fcn.caffemodel'
# init
#caffe.set_device(int(sys.argv[1]))

caffe.set_device(1)
caffe.set_mode_gpu()


solver = caffe.SGDSolver('solver.prototxt')
VGGnet=caffe.Net(proto,weights,caffe.TRAIN)
surgery.transplant(solver.net,VGGnet)
del VGGnet
#solver.net.copy_from(weights)
# surgeries
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
#surgery.interp(solver.net, interp_layers)

# scoring
solver.step(60000)
    
