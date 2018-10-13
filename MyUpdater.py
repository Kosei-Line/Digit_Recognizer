import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import math
import six
from chainer import cuda, training, reporter, Variable
from chainer.training import trainer, extensions
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import optimizer as optimizer_module


class MyUpdater(training.StandardUpdater):
    def __init__(self, iterator, CLS, opt,
        converter=convert.concat_examples,device=0):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main':iterator}
        self._iterators = iterator
        self.CLS = CLS
        self._optimizers = {"main":opt}
        self.converter = convert.concat_examples
        self.device = device
        self.iteration = 0

    def update_core(self):
        """lossを計算"""
        iterator = self._iterators['main'].next()
        input = self.converter(iterator, self.device)
        xp = np if int(self.device) == -1 else cuda.cupy
        x_batch = xp.array(input[0])
        t_batch = xp.array(input[1])
        self.loss = 0
        self.acc = 0
        #計算開始
        y = self.CLS(x_batch, softmax=False)
        self.loss = F.softmax_cross_entropy(y, t_batch)
        self.acc = F.accuracy(y, t_batch)
        self._optimizers["main"].target.cleargrads()
        self.loss.backward()
        self._optimizers["main"].update()
        reporter.report({'main/loss':self.loss, 'main/acc':self.acc})
