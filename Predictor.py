import chainer
import numpy as np
import pandas as pd
from chainer import serializers
import matplotlib.pyplot as plt
from chainer.datasets import tuple_dataset

import Mod
args = Mod.args
Net = Mod.Net

def Load_Dataset(withlabel=False, conv=True):
    input = pd.read_csv("test.csv")
    data = input.values
    #画像とラベルに分ける．画像は255で割って0~1に正規化
    img = (data/255.0).astype(np.float32)
    #imgを28x28にreshape
    if conv:
        img = np.reshape(img, (-1, 1, 28, 28))
    #tupledatasetを作る
    test = tuple_dataset.TupleDataset(img,)
    return test

def Calc_accuracy():
    CLS = Net.CLS()
    #load model
    serializers.load_npz('result/b{}/CLS_epoch_{}'.format(args.batchsize,
    args.epoch), CLS)
    #dataset
    _,test = chainer.datasets.get_mnist(withlabel=True, ndim=3)
    success = 0
    fail = 0
    #calc
    for i in range(1000):
        x,t = test[i]
        CLS.to_cpu()
        with chainer.using_config('train', False), \
            chainer.using_config('enable_backprop', False):
            y = CLS(x[None, ...]).data.argmax(axis=1)[0]
        #z = model.predictor(x[None, ...]).data#softmax全体を表示
        if(y==t):
            success+=1
        else:
            fail+=1
        #progress
        if(i%100==0):
            print(i)
    #out
    print("result: success={0}, fail={1}".format(success,fail))
    print("ratio = ",success/1000)

def Predict_num():
    CLS = Net.CLS()
    serializers.load_npz('result/b{}/CLS_epoch_{}'.format(args.batchsize,
    args.epoch), CLS)
    test = Load_Dataset()
    test_labels = np.zeros(1).astype(np.int32)
    for i in range(0, 28000):
        x = test[i][0]
        plt.imshow(x.reshape(28,28), cmap='gray')
        plt.show()
        with chainer.using_config('train', False), \
            chainer.using_config('enable_backprop', False):
            y = CLS(x[None, ...]).data.argmax(axis=1)[0]
        test_labels = np.append(test_labels, y)
        if(i%1000==0):
            print(i)
        print("predict:", y)
    test_labels = test_labels[1:]
    return test_labels

Calc_accuracy()
"""
test_labels = Predict_num()
submmission = pd.DataFrame(data={'ImageId':(np.arange(test_labels.shape[0])+1),
    'Label':test_labels})
submmission.to_csv('submission.csv', index=False)
submmission.tail()
"""
