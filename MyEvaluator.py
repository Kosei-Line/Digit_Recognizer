import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import cuda, training, reporter, Variable
from chainer.training import trainer, extensions
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import optimizer as optimizer_module
import copy

class MyEvaluator(extensions.Evaluator):
    def __init__(self, iterator, CLS,
        converter=convert.concat_examples,device=0, eval_hook=None,
        eval_func=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self._targets = {'CLS':CLS}

        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook

    def evaluate(self):
        iterator = self._iterators['main']
        self.CLS = self._targets['CLS']
        xp = np if int(self.device) == -1 else cuda.cupy
        it = copy.copy(iterator)
        summary = reporter.DictSummary()
        for batch in it:
            observation = {}
            with reporter.report_scope(observation):
                input = self.converter(batch, self.device)
                x_batch = xp.array(input[0])
                t_batch = xp.array(input[1])
                self.loss = 0
                self.acc = 0
                with chainer.using_config('train', False), \
                        chainer.using_config('enable_backprop', False):
                            y = self.CLS(x_batch, softmax=False)
                            self.loss = F.softmax_cross_entropy(y, t_batch)
                            self.acc = F.accuracy(y, t_batch)
                observation['val/loss'] = self.loss
                observation['val/acc'] = self.acc
            summary.add(observation)
        return summary.compute_mean()
