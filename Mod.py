import argparse

parser = argparse.ArgumentParser(description='MNIST')
parser.add_argument('--init', '-i', default=0, type=int)
parser.add_argument('--gpu', '-g', default=0, type=int)
parser.add_argument('--out', '-o', default='result')
parser.add_argument('--epoch', '-e', default=100, type=int)
parser.add_argument('--batchsize', '-b', default=100, type=int)
parser.add_argument('--file_name', '-f', default='test', type=str)
parser.add_argument('--snapshot', '-s', default=10, type=int)
parser.add_argument('--withlabel', '-wl', default=False, type=bool)

args = parser.parse_args()

print('#GPU:{}'.format(args.gpu))
print('#minibatch-size:{}'.format(args.batchsize))
print('#epoch:{}'.format(args.epoch))
print('')


import MyNet as Net
import MyUpdater as Updater
import MyEvaluator as Evaluator
