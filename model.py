import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import conv1d, mse_loss
import musicnet
import numpy as np
import copy

def create_filters(d,k,low=50,high=6000):
    x = np.linspace(0, 2*np.pi, d, endpoint=False)
    wsin = np.empty((k,1,d), dtype=np.float32)
    wcos = np.empty((k,1,d), dtype=np.float32)
    start_freq = low
    end_freq = high
    num_cycles = start_freq*d/44000.
    scaling_ind = np.log(end_freq/start_freq)/k
    window_mask = 1.0-1.0*np.cos(x)
    for ind in range(k):
        wsin[ind,0,:] = window_mask*np.sin(np.exp(ind*scaling_ind)*num_cycles*x)
        wcos[ind,0,:] = window_mask*np.cos(np.exp(ind*scaling_ind)*num_cycles*x)
    
    return wsin,wcos


class Model(torch.nn.Module):
    def __init__(self, avg=.9998,stride=512,regions=25,d=4096,k=500,m=128):
        super(Model, self).__init__()
        
        wsin,wcos = create_filters(d,k)
        with torch.cuda.device(0):
            self.wsin_var = Variable(torch.from_numpy(wsin).cuda(), requires_grad=False)
            self.wcos_var = Variable(torch.from_numpy(wcos).cuda(), requires_grad=False)
        
        self.linear = torch.nn.Linear(regions*k, m, bias=False).cuda()
        torch.nn.init.constant(self.linear.weight, 0)

        self.stride=stride
        self.regions=regions
        self.k=k


        self.avg = avg
        self.averages = copy.deepcopy(list(parm.data for parm in self.parameters()))
        for (name,parm),pavg in zip(self.named_parameters(),self.averages):
            name = name.split('.')[0]
            self.register_buffer(name + '_avg', pavg)
    
    def forward(self, x):
        zx = conv1d(x[:,None,:], self.wsin_var, stride=self.stride).pow(2) \
           + conv1d(x[:,None,:], self.wcos_var, stride=self.stride).pow(2)
        return self.linear(torch.log(zx + musicnet.epsilon).view(x.data.size()[0],self.regions*self.k))
    
    def average_iterates(self):
        for parm, pavg in zip(self.parameters(), self.averages):
            pavg.mul_(self.avg).add_(1.-self.avg, parm.data)

