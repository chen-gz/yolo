import os
import python.darknet as dn

dn.set_gpu(0)
net = dn.load_net(str.encode('cfg/yolov3.cfg'),
                  str.encode('weights/yolov3.weights'), 0)

meta = dn.load_meta(str.encode('cfg/coco.data'))
#  r = dn.detect(net, meta, str.encode('data/dog.jpg'))

#  print(r)
images = []
for root, dirs, files in os.walk("./africa", topdown=False):
    for name in files:
        images.append(os.path.join(root, name))
        print(os.path.join(root, name))
for i in images:
    r = dn.detect(net, meta, str.encode(i))
    print(r)
