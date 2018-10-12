import chainer
import chainer.functions as F
import chainer.links as L

import Mod
args = Mod.args

class CLS(chainer.Chain):
    #入力は12チャネル白黒画像の塊
    def __init__(self):
        super(CLS, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 8, 3, pad=1)
            self.ebn1 = L.BatchNormalization(8)
            self.conv2 = L.Convolution2D(8, 16, 3, pad=1)
            self.ebn2 = L.BatchNormalization(16)
            self.conv3 = L.Convolution2D(16, 32, 3, pad=1)
            self.fc1 = L.Linear(None, 1000)
            self.fc2 = L.Linear(1000, 10)

    def __call__(self, x, softmax=True):
        return self.cls(x, softmax)
    def cls(self, x, softmax=True):
        h = F.leaky_relu(self.ebn1(self.conv1(x)))
        h = F.leaky_relu(self.ebn2(self.conv2(h)))
        h = F.leaky_relu(self.conv3(h))
        h = F.leaky_relu(self.fc1(h))
        h = F.leaky_relu(self.fc2(h))
        if softmax:
            return F.softmax(h)
        else:
            return h
