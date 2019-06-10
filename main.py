import sys,os,errno,signal,copy
from contextlib import contextmanager

import numpy as np
import musicnet2 as musicnet

import torch
from torch.autograd import Variable
from torch.nn.functional import conv1d, mse_loss,l1_loss

from time import time

import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score
import argparse
from model import Baseline,Model

parser = argparse.ArgumentParser()
parser.add_argument('--mmap',default=1,type=int)
parser.add_argument('--multi',default=1,type=int)
parser.add_argument('--batch_size',default=100,type=int)
parser.add_argument('--mode', default='hybrid',type=str,choices=['hybrid','instruments','notes'])
parser.add_argument('--data_reload',default=0,type=int,choices=[0,1])
parser.add_argument('--stft_size',default= 512)
parser.add_argument('--visible_device',default='1',type=str)
parser.add_argument('--lr',default=0.001,type=float)
parser.add_argument('--mm',default=0.95,type=float)
parser.add_argument('--optim',default='SGD',type=str)
parser.add_argument('--basechannel',default=16,type=int)
parser.add_argument('--l1norm',default=0.,type=float)
args = parser.parse_args()

root = './musicnet'
checkpoint_path = './checkpoints'
checkpoint = 'musicnet_' + args.mode + '.pt'
try:
    os.makedirs(checkpoint_path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES']=args.visible_device
def worker_init(args):
    signal.signal(signal.SIGINT, signal.SIG_IGN) # ignore signals so parent can handle them
    np.random.seed(os.getpid() ^ int(time())) # approximately random seed for workers

kwargs = {'num_workers': 4, 'pin_memory': True, 'worker_init_fn': worker_init} if args.multi==1 else {}

a = 128 if args.mode in ['hybrid','notes'] else 0
b = 11 if args.mode in ['hybrid','instruments'] else 0

m = a + b
k = 500 
d = 4096
window = 16384
stride = 512
batch_size = args.batch_size

regions = int(1 + (window - d)/stride)

mmap=True if args.mmap==1 else False
train_set = musicnet.MusicNet(root=root, train=True, window=window, mmap=mmap,m=m)#, pitch_shift=5, jitter=.1)
test_set = musicnet.MusicNet(root=root, train=False, window=window, epoch_size=5000,mmap = mmap,m=m)
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,**kwargs)
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,**kwargs)


def L(y_hat, y):
    # adjust for per-frame loss
    return mse_loss(y_hat, y)*128/2. + args.l1norm * l1_loss(y_hat,y)

@contextmanager
def averages(model):
    orig_parms = copy.deepcopy(list(parm.data for parm in model.parameters()))
    for parm, pavg in zip(model.parameters(), model.averages):
        parm.data.copy_(pavg)
    yield
    for parm, orig in zip(model.parameters(), orig_parms):
        parm.data.copy_(orig)

model = Model(m=m,basechannel = args.basechannel).cuda()
print(model)
loss_history = []
avgp_history = []
if args.data_reload==1:
    try:
        model.load_state_dict(torch.load(os.path.join(checkpoint_path,checkpoint)))
    except IOError as e:
        if e.errno != errno.ENOENT:
            raise

optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=args.mm)
if args.optim=='Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

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
