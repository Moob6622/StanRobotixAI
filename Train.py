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

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation



class DataFeeder (object) :
    def __init__ (self, coder, size, mean) :

        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__ (self, path, index) :
        
        bboxDat = np.load(path + '/bboxs.npz')
        lblDat  = np.load(path + '/lbls.npz')

        imgPath  = bboxDat.files[index]
        img      = cv2.imread('./imgs/'+ imgPath +'.jpg')
        bbox     = bboxDat[img]
        lbl      = lblDat[img]

        # 1. Augmentation de couleur
        img = random_distort(img)

        # 2. Expansion aléatoire

        if np.random.randint(2) :
            img, param = transform.random_expand(
                img, fill=self.mean, return_param=True)
            
            bbox = transform.translate_bbox(
                bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

        # 3. Coupage aléatoir

        img, param = random_crop_with_bbox_constraints(
            img, bbox, return_param=True)
        
        bbox, param = transforms.crop_bbox(
            bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
            allow_outside_center=False, return_param=True)
        
        label = label[param['index']]

        # 4. Redimensionner avec interpolation aléatoire

        _, H, W = img.shape
        
        img = resize_with_random_interpolation(img, (self.size, self.size))
        
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. Random horizontal flipping
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        
        bbox = transforms.flip_bbox(
            bbox, (self.size, self.size), x_flip=params['x_flip'])

        # Formatation pour le réseaux SSD
        img -= self.mean
        mb_loc, mb_label = self.coder.encode(bbox, label)

        return img, mb_loc, mb_label
        
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default = './data')
    parser.add_argument('--basesize', type = int)
    parser.add_argument('--size', type = int)
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume')
    args = parser.parse_args()

    if args.model == 'ssd300':
        model = SSD300(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model='imagenet')
    elif args.model == 'ssd512':
        model = SSD512(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model='imagenet')

    model.use_preset('evaluate')
    
    train_chain = MultiboxTrainChain(model)
    
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()
    
    feeder = DataFeeder(model.coder, model.insize, model.mean)
    
    dataset = [feeder(args.path, i%args.basesize) for i in range (0, args.size)]
        





        
