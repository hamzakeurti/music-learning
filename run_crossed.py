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
from model import Baseline,CrossStitchModel

parser = argparse.ArgumentParser()
parser.add_argument('--mmap',default=1,type=int)
parser.add_argument('--multi',default=1,type=int)
parser.add_argument('--batch_size',default=100,type=int)
parser.add_argument('--data_reload',default=0,type=int,choices=[0,1])
parser.add_argument('--visible_device',default='1',type=str)
parser.add_argument('--lr',default=0.0001,type=float)
parser.add_argument('--mm',default=0.95,type=float)
parser.add_argument('--optim',default='SGD',type=str)
parser.add_argument('--basechannel',default=16,type=int)
parser.add_argument('--l1norm',default=0.,type=float)
parser.add_argument('--stitch_levels',nargs='*',type=int,default = [1,2])
parser.add_argument('--epochs',default=5,type=int)
parser.add_argument('--model',default='Baseline',type=str,choices = ['Baseline'])
args = parser.parse_args()

mode = 'hybrid'
model_dict={'Baseline':Baseline}
root = '/data/valentin/music-learning/musicnet'
checkpoint_path = './checkpoints'
checkpoint_n = 'musicnet_'+args.model + '_' + mode + '_n.pt'
checkpoint_i = 'musicnet_'+args.model + '_' + mode + '_i.pt'
checkpoint_tot = 'musicnet_'+args.model + '_' + mode + '_tot.pt'


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

a = 128 if mode in ['hybrid','notes'] else 0
b = 11 if mode in ['hybrid','instruments'] else 0

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



model_n = model_dict[args.model](m=m,basechannel = args.basechannel).cuda()
model_i = model_dict[args.model](m=m,basechannel = args.basechannel).cuda()

print(model_n)
print(model_i)

loss_history = []
avgp_history = []

if args.data_reload==1:
    try:
        model_n.load_state_dict(torch.load(os.path.join(checkpoint_path,checkpoint_n)))
        model_i.load_state_dict(torch.load(os.path.join(checkpoint_path,checkpoint_i)))
    except IOError as e:
        if e.errno != errno.ENOENT:
            raise
optimizer_n = torch.optim.SGD(model_n.parameters(), lr = args.lr, momentum=args.mm)
if args.optim=='Adam':
    optimizer_n = torch.optim.Adam(model_n.parameters(), lr = args.lr)
optimizer_i = torch.optim.SGD(model_i.parameters(), lr = args.lr, momentum=args.mm)
if args.optim=='Adam':
    optimizer_i = torch.optim.Adam(model_i.parameters(), lr = args.lr)




def run_model(model,optimizer,task,checkpoint):
    l_history_i = []
    l_history_n = []
    l_history_tot = []
    avgp_history_tot = []
    avgp_history_i = []
    avgp_history_n = []
    try:
        with train_set, test_set:
            print('current task : training ' + task)
            print('square loss\tavg prec\tnote prec\tinstr prec\ttime\ttask')
            for epoch in range(args.epochs):
                t = time()

                for i, (x, y) in enumerate(train_loader):
                    x, y = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
                    pred = model(x)
                    loss_n = L(pred[:,:a],y[:,:a])
                    loss_i = L(pred[:,a:],y[:,a:])
                    optimizer.zero_grad()
                    if task == 'notes':
                        loss_n.backward()
                    if task == 'instru':
                        loss_i.backward()
                    if task == 'stitch':
                        (loss_i+loss_n).backward()
                    optimizer.step()
                    model.average_iterates()
                t1 = time()
                avgp_tot,avgp_n,avgp_i, l_i,l_n,l_tot = 0., 0., 0., 0., 0., 0.
                yground = torch.FloatTensor(batch_size*len(test_loader), m)
                yhat = torch.FloatTensor(batch_size*len(test_loader), m)
                with averages(model):
                    for i, (x, y) in enumerate(test_loader):
                        x, y = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
                        yhatvar = model(x)
                        l_n += L(yhatvar[:,:a],y[:,:a]).data.item()
                        l_i += L(yhatvar[:,a:],y[:,a:]).data.item()
                        l_tot += L(yhatvar,y).data.item()
                        yground[i*batch_size:(i+1)*batch_size,:] = y.data
                        yhat[i*batch_size:(i+1)*batch_size,:] = yhatvar.data
                avgp_tot = average_precision_score(yground.numpy().flatten(),yhat.numpy().flatten())
                avgp_n = average_precision_score(yground[:,:a].numpy().flatten(),yhat[:,:a].numpy().flatten())
                avgp_i = average_precision_score(yground[:,a:].numpy().flatten(),yhat[:,a:].numpy().flatten())

                l_history_i.append(l_i/len(test_loader))
                l_history_n.append(l_n/len(test_loader))
                l_history_tot.append(l_tot/len(test_loader))
                avgp_history_tot.append(avgp_tot)
                avgp_history_i.append(avgp_i)
                avgp_history_n.append(avgp_n)
                torch.save(model.state_dict(), os.path.join(checkpoint_path,checkpoint))
                print('{:.4f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.2f}\t'.format(l_history_tot[-1],avgp_history_tot[-1],avgp_history_n[-1],avgp_history_i[-1],time()-t)+task)

    except KeyboardInterrupt:
        print('Graceful Exit')
    else:
        print('Finished')


run_model(model_n,optimizer_n,task="notes",checkpoint=checkpoint_n)

run_model(model_i,optimizer_i,task="instru",checkpoint=checkpoint_i)



model_n.train(False)
model_i.train(False)
model_tot = CrossStitchModel(model_n=model_n,model_i = model_i,levels_to_stitch = args.stitch_levels)
optimizer_tot = torch.optim.SGD(model_tot.parameters(), lr = args.lr, momentum=args.mm)
if args.optim=='Adam':
    optimizer_tot = torch.optim.Adam(model_tot.parameters(), lr = args.lr)

run_model(model_tot,optimizer_tot,task="stitch",checkpoint=checkpoint_tot)


# l_history_i = []
# l_history_n = []
# l_history_tot = []
# avgp_history_tot = []
# avgp_history_i = []
# avgp_history_n = []
# try:
#     with train_set, test_set:
#         print('current task : training stitched network')
#         task = 'stitched'
#         print('task\tsquare loss\tavg prec\tnote prec\tinstr prec\ttime\t\tutime')
#         for epoch in range(args.epochs):
#             t = time()

#             for i, (x, y) in enumerate(train_loader):
#                 x, y = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
#                 pred = model_tot(x)
#                 loss_n = L(pred[:,:a],y[:,:a])
#                 loss_i = L(pred[:,a:],y[:,a:])
#                 optimizer_tot.zero_grad()
#                 (loss_i+loss_n).backward()
#                 optimizer_tot.step()
#                 model_tot.average_iterates()
#             t1 = time()
#             avgp_tot,avgp_n,avgp_i, loss_i,loss_n,loss_tot = 0., 0., 0., 0., 0., 0.
#             yground = torch.FloatTensor(batch_size*len(test_loader), m)
#             yhat = torch.FloatTensor(batch_size*len(test_loader), m)
#             with averages(model_tot):
#                 for i, (x, y) in enumerate(test_loader):
#                     x, y = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
#                     yhatvar = model_tot(x)
#                     loss_n += L(yhatvar[:,:a],y[:,:a]).data.item()
#                     loss_i += L(yhatvar[:,a:],y[:,a:]).data.item()
#                     loss_tot += L(yhatvar,y).data.item()
#                     yground[i*batch_size:(i+1)*batch_size,:] = y.data
#                     yhat[i*batch_size:(i+1)*batch_size,:] = yhatvar.data
#             avgp_tot = average_precision_score(yground.numpy().flatten(),yhat.numpy().flatten())
#             avgp_n = average_precision_score(yground[:,:a].numpy().flatten(),yhat[:,:a].numpy().flatten())
#             avgp_i = average_precision_score(yground[:,a:].numpy().flatten(),yhat[:,a:].numpy().flatten())

#             loss_history_i.append(loss_i/len(test_loader))
#             loss_history_n.append(loss_n/len(test_loader))
#             loss_history_tot.append(loss_tot/len(test_loader))
#             avgp_history_tot.append(avgp_tot)
#             avgp_history_i.append(avgp_i)
#             avgp_history_n.append(avgp_n)
#             torch.save(model_tot.state_dict(), os.path.join(checkpoint_path,checkpoint_i))
#             print(task+'\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}'.format(loss_history_tot[-1],avgp_history_tot[-1],avgp_history_n[-1],avgp_history_i[-1],time()-t, time()-t1))



# except KeyboardInterrupt:
#     print('Graceful Exit')
# else:
#     print('Finished')


# loss_history_i = []
# loss_history_n = []
# loss_history_tot = []
# avgp_history_tot = []
# avgp_history_i = []
# avgp_history_n = []


# try:
#     with train_set, test_set:
#         task = 'notes'
#         print('current task : training notes')
#         print('task\tsquare loss\tavg prec\tnote prec\tinstr prec\ttime\t\tutime')
#         for epoch in range(args.epochs):
#             t = time()
#             train_n = epoch % 2

#             for i, (x, y) in enumerate(train_loader):
#                 x, y = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
#                 pred = model_n(x)
#                 loss_n = L(pred[:,:a],y[:,:a])
#                 loss_i = L(pred[:,a:],y[:,a:])
#                 optimizer_n.zero_grad()
#                 loss_n.backward()
#                 optimizer_n.step()
#                 model_n.average_iterates()
#             t1 = time()
#             avgp_tot,avgp_n,avgp_i, loss_i,loss_n,loss_tot = 0., 0., 0., 0., 0., 0.
#             yground = torch.FloatTensor(batch_size*len(test_loader), m)
#             yhat = torch.FloatTensor(batch_size*len(test_loader), m)
#             with averages(model_n):
#                 for i, (x, y) in enumerate(test_loader):
#                     x, y = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
#                     yhatvar = model_n(x)
#                     loss_n += L(yhatvar[:,:a],y[:,:a]).data.item()
#                     loss_i += L(yhatvar[:,a:],y[:,a:]).data.item()
#                     loss_tot += L(yhatvar,y).data.item()
#                     yground[i*batch_size:(i+1)*batch_size,:] = y.data
#                     yhat[i*batch_size:(i+1)*batch_size,:] = yhatvar.data
#             avgp_tot = average_precision_score(yground.numpy().flatten(),yhat.numpy().flatten())
#             avgp_n = average_precision_score(yground[:,:a].numpy().flatten(),yhat[:,:a].numpy().flatten())
#             avgp_i = average_precision_score(yground[:,a:].numpy().flatten(),yhat[:,a:].numpy().flatten())

#             loss_history_i.append(loss_i/len(test_loader))
#             loss_history_n.append(loss_n/len(test_loader))
#             loss_history_tot.append(loss_tot/len(test_loader))
#             avgp_history_tot.append(avgp_tot)
#             avgp_history_i.append(avgp_i)
#             avgp_history_n.append(avgp_n)
#             torch.save(model_n.state_dict(), os.path.join(checkpoint_path,checkpoint_n))
#             print('\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}'.format(loss_history_tot[-1],avgp_history_tot[-1],avgp_history_n[-1],avgp_history_i[-1],time()-t, time()-t1))

# except KeyboardInterrupt:
#     print('Graceful Exit')
# else:
#     print('Finished')