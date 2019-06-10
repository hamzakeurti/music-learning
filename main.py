import sys,os,errno,signal,copy
from contextlib import contextmanager

import numpy as np
import musicnet

import torch
from torch.autograd import Variable
from torch.nn.functional import conv1d, mse_loss

from time import time

import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score
import argparse
from model import Baseline

parser = argparse.ArgumentParser()
parser.add_argument('--mmap',default=0,type=int)
parser.add_argument('--multi',default=0,type=int)
parser.add_argument('--batch_size',default=100,type=int)
args = parser.parse_args()

root = './musicnet'
checkpoint_path = './checkpoints'
checkpoint = 'musicnet_demo.pt'

try:
    os.makedirs(checkpoint_path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES']='1'
def worker_init(args):
    signal.signal(signal.SIGINT, signal.SIG_IGN) # ignore signals so parent can handle them
    np.random.seed(os.getpid() ^ int(time())) # approximately random seed for workers

kwargs = {'num_workers': 4, 'pin_memory': True, 'worker_init_fn': worker_init} if args.multi==1 else {}

m = 128
k = 500
d = 4096
window = 16384
stride = 512
batch_size = args.batch_size

regions = int(1 + (window - d)/stride)

mmap=True if args.mmap==1 else False
train_set = musicnet.MusicNet(root=root, train=True, window=window, mmap=mmap)#, pitch_shift=5, jitter=.1)
test_set = musicnet.MusicNet(root=root, train=False, window=window, epoch_size=5000,mmap = mmap)
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,**kwargs)
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,**kwargs)


def L(y_hat, y):
    # adjust for per-frame loss
    return mse_loss(y_hat, y)*128/2.

@contextmanager
def averages(model):
    orig_parms = copy.deepcopy(list(parm.data for parm in model.parameters()))
    for parm, pavg in zip(model.parameters(), model.averages):
        parm.data.copy_(pavg)
    yield
    for parm, orig in zip(model.parameters(), orig_parms):
        parm.data.copy_(orig)

model = Baseline()
print(model)
loss_history = []
avgp_history = []

try:
    model.load_state_dict(torch.load(os.path.join(checkpoint_path,checkpoint)))
except IOError as e:
    if e.errno != errno.ENOENT:
        raise
optimizer = torch.optim.SGD(model.parameters(), lr=0.000001, momentum=.95)

try:
    with train_set, test_set:
        print('square loss\tavg prec\ttime\t\tutime')
        for epoch in range(50):
            t = time()
            for i, (x, y) in enumerate(train_loader):
                x, y = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
                loss = L(model(x),y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.average_iterates()
            t1 = time()
            avgp, loss = 0., 0.
            yground = torch.FloatTensor(batch_size*len(test_loader), m)
            yhat = torch.FloatTensor(batch_size*len(test_loader), m)
            with averages(model):
                for i, (x, y) in enumerate(test_loader):
                    x, y = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
                    yhatvar = model(x)
                    loss += L(yhatvar,y).data.item()
                    yground[i*batch_size:(i+1)*batch_size,:] = y.data
                    yhat[i*batch_size:(i+1)*batch_size,:] = yhatvar.data
            avgp = average_precision_score(yground.numpy().flatten(),yhat.numpy().flatten())
            loss_history.append(loss/len(test_loader))
            avgp_history.append(avgp)
            torch.save(model.state_dict(), os.path.join(checkpoint_path,checkpoint))
            print('{:2f}\t{:2f}\t{:2f}\t{:2f}'.format(loss_history[-1],avgp_history[-1],time()-t, time()-t1))

except KeyboardInterrupt:
    print('Graceful Exit')
else:
    print('Finished')