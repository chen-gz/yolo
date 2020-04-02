import python.darknet as dn

dn.set_gpu(0)
net = dn.load_net('cfg/tiny-yolo.cfg', 'weights/tiny-yolo.weights', 0)

meta = dn.load_meta('cfg/coco.data')
r = dn.detect(net, meta, 'data/dog.jpg')
print(r)
