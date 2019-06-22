import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import conv1d, mse_loss
import musicnet
import numpy as np
import copy
import math
import torch.nn.functional as F

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


class NaiveFilter(torch.nn.Module):
    def __init__(self, avg=.9998,stride=512,regions=25,d=4096,k=500,m=128):
        super(NaiveFilter, self).__init__()
        
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

class Baseline(torch.nn.Module):
    def __init__(self, avg=.9998,stride=512,regions=25,d=4096,k=500,m=128,stft=512,window=16384,batch_size=100,basechannel=128):
        super(Baseline, self).__init__()

        self.stride=stride
        self.regions=regions
        self.k=k
        
        wsin,wcos = create_filters(d,k)
        with torch.cuda.device(0):
            self.wsin_var = Variable(torch.from_numpy(wsin).cuda(), requires_grad=False)
            self.wcos_var = Variable(torch.from_numpy(wcos).cuda(), requires_grad=False)
        #For stft
        self.stft=stft
        self.N = stft//2 + 1
        self.T = window//(4*stft) + 1

        self.avg = avg
        self.averages = copy.deepcopy(list(parm.data for parm in self.parameters()))
        for (name,parm),pavg in zip(self.named_parameters(),self.averages):
            name = name.split('.')[0]
            self.register_buffer(name + '_avg', pavg)

        self.batch_size=batch_size
        self.inshape=251*2*basechannel
        self.conv1 = nn.Conv2d(1,basechannel,(128,1),stride=(2,1),padding=(64,0))
        self.conv2 = nn.Conv2d(basechannel,2*basechannel,(1,25))
        self.linear = nn.Linear(self.inshape,m)
        self.norm1 = nn.BatchNorm2d(basechannel)
        self.norm2 = nn.BatchNorm2d(basechannel*2)
    def forward(self, x):
        zx = conv1d(x[:,None,:], self.wsin_var, stride=self.stride).pow(2) \
           + conv1d(x[:,None,:], self.wcos_var, stride=self.stride).pow(2)
        #batch size * 500 * 25
        zx = zx.unsqueeze(1)
        x = F.relu(self.conv1(torch.log(zx + 10e-15)))
        # batch size *basechannel * 501 * 25
        x = self.norm1(x)
        x =  F.relu(self.conv2(x))
        x = self.norm2(x)
        # batchsize * basechannel2 * 501 * 1
        x = x.reshape(self.batch_size,self.inshape)
        return self.linear(x)
    def average_iterates(self):
        for parm, pavg in zip(self.parameters(), self.averages):
            pavg.mul_(self.avg).add_(1.-self.avg, parm.data)

class HardParameterSharing1(torch.nn.Module):
    def __init__(self, avg=.9998,stride=512,regions=25,d=4096,k=500,m=139,stft=512,window=16384,batch_size=100,basechannel=128):
        super(HardParameterSharing1, self).__init__()

        self.stride=stride
        self.regions=regions
        self.k=k
        
        wsin,wcos = create_filters(d,k)
        with torch.cuda.device(0):
            self.wsin_var = Variable(torch.from_numpy(wsin).cuda(), requires_grad=False)
            self.wcos_var = Variable(torch.from_numpy(wcos).cuda(), requires_grad=False)
        #For stft
        self.stft=stft
        self.N = stft//2 + 1
        self.T = window//(4*stft) + 1

        self.avg = avg
        self.averages = copy.deepcopy(list(parm.data for parm in self.parameters()))
        for (name,parm),pavg in zip(self.named_parameters(),self.averages):
            name = name.split('.')[0]
            self.register_buffer(name + '_avg', pavg)

        self.batch_size=batch_size
        self.inshape=251*2*basechannel
        self.conv1 = nn.Conv2d(1,basechannel,(128,1),stride=(2,1),padding=(64,0))
        self.conv2 = nn.Conv2d(basechannel,2*basechannel,(1,25))
        self.linear_note = nn.Linear(self.inshape,128)
        self.linear_inst = nn.Linear(self.inshape,11)
        
        self.norm1 = nn.BatchNorm2d(basechannel)
        self.norm2 = nn.BatchNorm2d(basechannel*2)
    def forward(self, x):
        #Shared part
        zx = conv1d(x[:,None,:], self.wsin_var, stride=self.stride).pow(2) \
           + conv1d(x[:,None,:], self.wcos_var, stride=self.stride).pow(2)
        zx = zx.unsqueeze(1)
        shared = F.relu(self.conv1(torch.log(zx + 10e-15)))
        shared = self.norm1(shared)

        #Disjonction
        x_note =  F.relu(self.conv2(shared))
        x_note = self.norm2(x_note)
        x_note = x_note.reshape(self.batch_size,self.inshape)
        x_note = self.linear_note(x_note)

        x_inst =  F.relu(self.conv2(shared))
        x_inst = self.norm2(x_inst)
        x_inst = x_inst.reshape(self.batch_size,self.inshape)
        x_inst = self.linear_inst(x_inst)

        return torch.cat((x_note,x_inst),dim=1)

    def average_iterates(self):
        for parm, pavg in zip(self.parameters(), self.averages):
            pavg.mul_(self.avg).add_(1.-self.avg, parm.data)

class SoftParameterSharing(torch.nn.Module):
    def __init__(self, avg=.9998,stride=512,regions=25,d=4096,k=500,m=139,stft=512,window=16384,batch_size=100,basechannel=128):
        super(SoftParameterSharing, self).__init__()

        self.stride=stride
        self.regions=regions
        self.k=k
        
        wsin,wcos = create_filters(d,k)
        with torch.cuda.device(0):
            self.wsin_var = Variable(torch.from_numpy(wsin).cuda(), requires_grad=False)
            self.wcos_var = Variable(torch.from_numpy(wcos).cuda(), requires_grad=False)
        #For stft
        self.stft=stft
        self.N = stft//2 + 1
        self.T = window//(4*stft) + 1

        self.avg = avg
        self.averages = copy.deepcopy(list(parm.data for parm in self.parameters()))
        for (name,parm),pavg in zip(self.named_parameters(),self.averages):
            name = name.split('.')[0]
            self.register_buffer(name + '_avg', pavg)

        self.batch_size=batch_size
        self.inshape=251*2*basechannel
        # Soft shared parameters
        self.conv1_note = nn.Conv2d(1,basechannel,(128,1),stride=(2,1),padding=(64,0))
        self.conv1_inst = nn.Conv2d(1,basechannel,(128,1),stride=(2,1),padding=(64,0))

        self.conv2_note = nn.Conv2d(basechannel,2*basechannel,(1,25))
        self.conv2_inst = nn.Conv2d(basechannel,2*basechannel,(1,25))


        self.linear_note = nn.Linear(self.inshape,128)
        self.linear_inst = nn.Linear(self.inshape,11)
        
        self.norm1_note = nn.BatchNorm2d(basechannel)
        self.norm1_inst = nn.BatchNorm2d(basechannel)

        self.norm2_note = nn.BatchNorm2d(basechannel*2)
        self.norm2_inst = nn.BatchNorm2d(basechannel*2)

    def forward(self, x):
        zx = conv1d(x[:,None,:], self.wsin_var, stride=self.stride).pow(2) \
           + conv1d(x[:,None,:], self.wcos_var, stride=self.stride).pow(2)
        #batch size * 500 * 25
        zx = zx.unsqueeze(1)
        zx = torch.log(zx + 10e-15)


        x_note = F.relu(self.conv1_note(zx))
        x_note = self.norm1_note(x_note)

        x_inst = F.relu(self.conv1_inst(zx))
        x_inst = self.norm1_inst(x_inst)

        x_note =  F.relu(self.conv2_note(x_note))
        x_note = self.norm2_note(x_note)
        x_note = x_note.reshape(self.batch_size,self.inshape)
        x_note = self.linear_note(x_note)


        x_inst =  F.relu(self.conv2_inst(x_inst))
        x_inst = self.norm2_inst(x_inst)
        x_inst = x_inst.reshape(self.batch_size,self.inshape)
        x_inst = self.linear_inst(x_inst)

        return torch.cat((x_note,x_inst),dim=1)

    def average_iterates(self):
        for parm, pavg in zip(self.parameters(), self.averages):
            pavg.mul_(self.avg).add_(1.-self.avg, parm.data)



class StitchUnit(torch.nn.Module):
    def __init__(self):
        super(StitchUnit,self).__init__()
        self.stitch_matrix = nn.Parameter(data=torch.eye(2))

    def forward(self,x1,x2):
        x1 = (self.stitch_matrix[0,0])*x1 + self.stitch_matrix[1,0]*x2)/(self.stitch_matrix[0,0]) + self.stitch_matrix[1,0])
        x2 = self.stitch_matrix[0,1]*x1 + self.stitch_matrix[1,1]*x2/(self.stitch_matrix[0,1]) + self.stitch_matrix[1,1])
        return x1,x2
    


class CrossStitchModel(torch.nn.Module):
    def __init__(self,model_n,model_i,levels_to_stitch,avg=.9998):
        super(CrossStitchModel,self).__init__()
        self.model_n = model_n
        self.model_i = model_i
        # self.cross_matrix = Variable(torch.stack([torch.eye(2),torch.eye(2)])
        self.stitch_unit1, self.stitch_unit2 = StitchUnit().cuda(),StitchUnit().cuda()

        if 1 not in levels_to_stitch:
            self.stitch_unit1.train(False)
        if 2 not in levels_to_stitch:
            self.stitch_unit2.train(False)


        self.avg = avg
        self.averages = copy.deepcopy(list(parm.data for parm in self.parameters()))
        for (name,parm),pavg in zip(self.named_parameters(),self.averages):
            name = name.split('.')[0]
            self.register_buffer(name + '_avg', pavg)
        

    def forward(self,x):
        zx = conv1d(x[:,None,:], self.model_i.wsin_var, stride=self.model_i.stride).pow(2) \
           + conv1d(x[:,None,:], self.model_i.wcos_var, stride=self.model_i.stride).pow(2)
                   #batch size * 500 * 25
        zx = zx.unsqueeze(1)
        
        x1 = F.relu(self.model_n.conv1(torch.log(zx + 10e-15)))
        x2 = F.relu(self.model_i.conv1(torch.log(zx + 10e-15)))
        
        x1,x2 = self.stitch_unit1(x1,x2)

        # batch size *basechannel * 501 * 25
        x1 = self.model_n.norm1(x1)
        x1 =  F.relu(self.model_n.conv2(x1))

        x2 = self.model_i.norm1(x2)
        x2 =  F.relu(self.model_i.conv2(x2))


        # Crossing
        x1,x2 = self.stitch_unit2(x1,x2)

        

        # batchsize * basechannel2 * 501 * 1
        x1 = self.model_n.norm2(x1)
        x1 = x1.reshape(self.model_n.batch_size,self.model_n.inshape)

        # batch size *basechannel * 501 * 25

        # batchsize * basechannel2 * 502 * 2
        x2 = self.model_i.norm2(x2)
        x2 = x2.reshape(self.model_i.batch_size,self.model_i.inshape)
        return torch.cat((self.model_n.linear(x1)[:,:128], self.model_i.linear(x2)[:,128:]),dim=1)
    
    def average_iterates(self):
        for parm, pavg in zip(self.parameters(), self.averages):
            pavg.mul_(self.avg).add_(1.-self.avg, parm.data)