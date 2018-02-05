import argparse
import copy
import numpy as np
import cv2

import chainer
from chainer.datasets import ConcatenatedDataset
from chainer.datasets import TransformDataset
from chainer.optimizer import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import transforms
from chainercv import utils

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation


class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3):

        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report (
            {'loss': loss, 'loc/loss' : loc_loss, 'loss/conf' :conf_loss},
            self)
        return loss
    

class DataFeeder (object) :
    def __init__ (self, coder, size, mean) :

        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean
        

    def __call__ (self, path, index) :
        print('entered datafeeder')
        
        bboxDat = np.load(path + '/bboxs.npz')
        lblDat  = np.load(path + '/lbls.npz')

        imgPath  = bboxDat.files[index]
        img      = utils.read_image('./imgs/'+ imgPath +'.jpg', color=True)
        bbox     = bboxDat[imgPath].astype(np.float32)
        lbl      = lblDat[imgPath]

        
        print(utils.assert_is_image(img))
        # 1. Augmentation de couleur
        img = random_distort(img)
        print(utils.assert_is_image(img))

        # 2. Expansion aléatoire

        if np.random.randint(2) :
            img, param = transforms.random_expand(
                img, fill=self.mean, return_param=True)
            
            bbox = transforms.translate_bbox(
                bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])
        print(utils.assert_is_image(img))
        # 3. Coupage aléatoir

        img, param = random_crop_with_bbox_constraints(
            img, bbox, return_param=True)
        
        bbox, param = transforms.crop_bbox(
            bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
            allow_outside_center=False, return_param=True)
         
        lbl = lbl[param['index']]
        print(utils.assert_is_image(img))
        # 4. Redimensionner avec interpolation aléatoire

##        _, H, W = img.shape
##        
##        img = resize_with_random_interpolation(img, (self.size, self.size))
##        
##        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))
##        print(utils.assert_is_image(img))
        
        # 5. Random horizontal flipping
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        
        bbox = transforms.flip_bbox(
            bbox, (self.size, self.size), x_flip=params['x_flip'])
        print(utils.assert_is_image(img))
        # Formatation pour le réseaux SSD
##        img -= self.mean
        mb_loc, mb_label = self.coder.encode(bbox, lbl)
        print(utils.assert_is_image(img))
        return img, mb_loc, mb_label
        
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default = './data')
    parser.add_argument('--basesize', type = int, default = 3)
    parser.add_argument('--size', type = int, default = 20)
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')
    args = parser.parse_args()

    labelNames = ['cube']

    if args.model == 'ssd300':
        model = SSD300(
            n_fg_class=len(labelNames),
            pretrained_model='imagenet')
    elif args.model == 'ssd512':
        model = SSD512(
            n_fg_class=len(labelNames),
            pretrained_model='imagenet')

    model.use_preset('evaluate')
    
    trainChain = MultiboxTrainChain(model)
    
##    if args.gpu >= 0:
##        chainer.cuda.get_device_from_id(args.gpu).use()
##        model.to_gpu()
    
    feeder = DataFeeder(model.coder, model.insize, model.mean)
    
    #train = [feeder(args.path, i%args.basesize) for i in range (0, args.size)]
    train =[]
    for i in range (0, args.size) :
        print(i)
        datum = feeder(args.path, i%args.basesize)
        x,_,_ =datum
        utils.assert_is_image(x,color=True)
        train.append(datum)
        
    print (train[0])
    utils.assert_is_bbox_dataset(train, 1)
    utils.assert_is_label_dataset(train,1)
        
    trainIter = chainer.iterators.MultiprocessIterator (train, args.batchsize)

    #test  = [feeder(args.path, i%args.basesize) for i in range (0, args.size)]
    testArr =[]
    for i in range (0, args.size) :
        print(i)
        testArr.append(feeder(args.path, i%args.basesize))

    np.save('./data/test',testArr)#, allow_pickle = True)
    test = np.load('./data/test.npy')    
    testIter = chainer.iterators.SerialIterator(test, args.size, repeat=True, shuffle=True)


    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(trainChain)

    for param in trainChain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else :
            param.update_rule.add_hook(WeightDecay(0.0005))

    updater = training.StandardUpdater(trainIter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (120000, 'iteration'), args.out)
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=1e-3),
        trigger=triggers.ManualScheduleTrigger([80000, 100000], 'iteration'))

    trainer.extend(
        DetectionVOCEvaluator(
            testIter, model, use_07_metric=True,
            label_names=labelNames),
        trigger=(10000, 'iteration'))

    log_interval = 10, 'iteration'
    
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         'main/loss', 'main/loss/loc', 'main/loss/conf',
         'validation/main/map']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(extensions.snapshot(), trigger=(10000, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
        trigger=(120000, 'iteration'))

    trainer.run()

if __name__ == '__main__':
    main()
