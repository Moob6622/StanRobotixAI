import numpy as np
import chainer
import matplotlib
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class MyChain(Chain):
    def __init__(self, n1, n2,n3, nOut):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None,n1)
            self.l2 = L.Linear(None,n2)
            self.l3 = L.Linear(None,n3)
            self.l4 = L.Linear(None, nOut)
            
    def __call__ (self, x):

        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        y = self.l4(h3)
        return y
    
class Classifier(Chain):
    def __init__(self, predictor) :
        super(Classifier, self).__init__()
        with self.init_scope():
                self.predictor = predictor
                
    def __call__ (self, x, t):

        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        report({'loss' : loss, 'accuracy' : accuracy},self)
        return loss
        
        
    



train, test  = datasets.get_mnist()
train_iter = iterators.SerialIterator(train, batch_size = 100, shuffle = True)
test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

model = L.Classifier(MyChain(100,100,100,10))
optimiser = optimizers.SGD()
optimiser.setup(model)

updater = training.StandardUpdater(train_iter, optimiser)
trainer = training.Trainer(updater, (500, 'epoch'), out='result')
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))

trainer.run()

