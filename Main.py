import chainer
import numpy as np
import pandas as pd
from chainer import training
from chainer.datasets import tuple_dataset
from chainer.training import extensions

import Mod
args = Mod.args
Net = Mod.Net
Evaluator = Mod.Evaluator
Updater = Mod.Updater

def Load_Dataset(withlabel=True, conv=True):
    input = pd.read_csv("train.csv")
    data = input.values
    #画像とラベルに分ける．画像は255で割って0~1に正規化
    img = (data[:,1:]/255.0).astype(np.float32)
    label = data[:,:1].flatten()
    #imgを28x28にreshape
    if conv:
        img = np.reshape(img, (-1, 1, 28, 28))
    #trainとtestのしきい値を決める
    threshhold = np.int32(len(img)*0.8)
    #tupledatasetを作る
    if withlabel:
        train = tuple_dataset.TupleDataset(img[0:threshhold], label[0:threshhold])
        test = tuple_dataset.TupleDataset(img[threshhold:], label[threshhold:])
    else:
        train = tuple_dataset.TupleDataset(img[0:threshhold],)
        test = tuple_dataset.TupleDataset(img[threshhold:],)
    return train, test

def main():
    CLS = Net.CLS()
    CLS.to_gpu()
    print('Loading')
    train, test = Load_Dataset(withlabel=True, conv=True)
    #train, test = chainer.datasets.mnist.get_mnist(withlabel=True, ndim=3)
    print('Loaded')
    def make_optimizer(model, alpha=0.0002, beta1=0.9, beta2=0.999):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))
        return optimizer
    opt = make_optimizer(CLS)
    #Laod the dataset
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                repeat=False, shuffle=False)
    updater = Updater.MyUpdater(train_iter, CLS, opt, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'),
        out="{}/b{}".format(args.out, args.batchsize))
    trainer.extend(Evaluator.MyEvaluator(test_iter, CLS,
        device=args.gpu))
    trainer.extend(extensions.snapshot_object(CLS,
        filename="CLS_epoch_{.updater.epoch}"), trigger=(args.snapshot, 'epoch'))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/acc', 'val/loss',
        'val/acc', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()
    del trainer



if __name__ == '__main__':
    main()
