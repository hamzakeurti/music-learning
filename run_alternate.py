from time import time
import sys,os,errno,signal,copy
from contextlib import contextmanager

import argparse
import torch
from torch.autograd import Variable
from torch.nn.functional import conv1d, mse_loss,l1_loss
from sklearn.metrics import average_precision_score


import numpy as np
import musicnet2 as musicnet
from model import NaiveFilter,NaiveCNN,Baseline,ComplexModel




parser = argparse.ArgumentParser()
parser.add_argument('--mmap',default=1,type=int)
parser.add_argument('--multi',default=1,type=int)
parser.add_argument('--batch_size',default=100,type=int)
parser.add_argument('--data_reload',default=0,type=int,choices=[0,1])
parser.add_argument('--stft_size',default= 512)
parser.add_argument('--visible_device',default='1',type=str)
parser.add_argument('--lr',default=0.001,type=float)
parser.add_argument('--mm',default=0.95,type=float)
parser.add_argument('--optim',default='SGD',type=str)
parser.add_argument('--basechannel',default=16,type=int)
parser.add_argument('--l1norm',default=0.,type=float)
parser.add_argument('--epochs',default=50,type=int)
parser.add_argument('--model',default='Baseline',type=str,choices = ['NaiveFilter','NaiveCNN','Baseline','ComplexModel'])

args = parser.parse_args()
model_dict={'Baseline':Baseline,'NaiveCNN':NaiveCNN,'NaiveFilter':NaiveFilter,'ComplexModel':ComplexModel}
root = './musicnet'
checkpoint_path = './checkpoints'
checkpoint = 'musicnet_'+args.model + '_' + args.mode + '.pt'


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

a = 128
b = 11

m = 128+11
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
model = model_dict[args.model](m=m,basechannel = args.basechannel).cuda()
print(model)
loss_history_i = []
loss_history_n = []
loss_history_tot = []
avgp_history_tot = []
avgp_history_i = []
avgp_history_n = []

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
        print('square loss\tavg prec\tnote prec\tinstr prec\ttime\t\tutime')
        for epoch in range(args.epochs):
            t = time()
            train_n = epoch % 2
            for i, (x, y) in enumerate(train_loader):
                x, y = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
                pred = model(x)
                loss_n = L(pred[:,:a],y[:,:a])
                loss_i = L(pred[:,a:],y[:,a:])
                optimizer.zero_grad()
                if epoch:
                    loss_n.backward()
                else:
                    loss_i.backward()
                optimizer.step()
                model.average_iterates()
            t1 = time()
            avgp_tot,avgp_n,avgp_i, loss_i,loss_n,loss_tot = 0., 0., 0., 0.
            yground = torch.FloatTensor(batch_size*len(test_loader), m)
            yhat = torch.FloatTensor(batch_size*len(test_loader), m)
            with averages(model):
                for i, (x, y) in enumerate(test_loader):
                    x, y = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
                    yhatvar = model(x)
                    loss_n += L(yhatvar[:,:a],y[:,:a]).data.item()
                    loss_i += L(yhatvar[:,a:],y[:,a:]).data.item()
                    loss_tot += L(yhatvar,y).data.item()
                    yground[i*batch_size:(i+1)*batch_size,:] = y.data
                    yhat[i*batch_size:(i+1)*batch_size,:] = yhatvar.data
            avgp_tot = average_precision_score(yground.numpy().flatten(),yhat.numpy().flatten())
            avgp_n = average_precision_score(yground[:,:a].numpy().flatten(),yhat[:,:a].numpy().flatten())
            avgp_i = average_precision_score(yground[:,a:].numpy().flatten(),yhat[:,a:].numpy().flatten())

            loss_history_i.append(loss_i/len(test_loader))
            loss_history_n.append(loss_n/len(test_loader))
            loss_history_tot.append(loss_tot/len(test_loader))
            avgp_history_tot.append(avgp_tot)
            avgp_history_i.append(avgp_i)
            avgp_history_n.append(avgp_n)
            torch.save(model.state_dict(), os.path.join(checkpoint_path,checkpoint))
            print('{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}'.format(loss_history_tot[-1],avgp_history_tot[-1],avgp_history_n[-1],avgp_history_i[-1],time()-t, time()-t1))

except KeyboardInterrupt:
    print('Graceful Exit')
else:
    print('Finished')
