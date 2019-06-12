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

class NaiveCNN(torch.nn.Module):
    def __init__(self, avg=.9998,stride=512,regions=25,d=4096,k=500,m=128,stft=512,window=16384,batch_size=100,basechannel=16):
        super(NaiveCNN, self).__init__()

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

        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=3,stride=2)
        self.conv1 = nn.Conv2d(1,basechannel,3,padding=1)
        self.conv2 =nn.Conv2d(basechannel,basechannel*2,5,stride=2)
        self.conv3 = nn.Conv2d(basechannel*2,basechannel*4,3,padding=1)
        self.conv4 = nn.Conv2d(basechannel*4,basechannel*8,3,padding=1) 
        self.conv5 = nn.Conv2d(basechannel*8,basechannel*4,3,padding=1)
        self.inshape = 244*basechannel
        self.linear = nn.Linear(244*basechannel,m)
        self.norm1 = nn.BatchNorm2d(basechannel)
        self.norm2 = nn.BatchNorm2d(basechannel*2)
        self.norm3 = nn.BatchNorm2d(basechannel*4)
        self.norm4 = nn.BatchNorm2d(basechannel*8)
    def forward(self, x):
        zx = conv1d(x[:,None,:], self.wsin_var, stride=self.stride).pow(2) \
           + conv1d(x[:,None,:], self.wcos_var, stride=self.stride).pow(2)
        zx = zx.unsqueeze(1)
        x = F.relu(self.conv1(zx))
        x = self.norm1(x)
        x = self.maxpool(x)
        x =  F.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.norm3(x)
        x = x.reshape(self.batch_size,self.inshape)
        return self.linear(x)
    
    def average_iterates(self):
        for parm, pavg in zip(self.parameters(), self.averages):
            pavg.mul_(self.avg).add_(1.-self.avg, parm.data)


class ToneNN(torch.nn.Module):
    def __init__(self,conv1_channels,conv2_channels,conv3_channels,conv_size,pool_size,num_features,m=128):
        super(ToneNN,self).__init__()

        self.frequency_span = frequency_span
        self.time_span = time_span
        
        self.pool_size = pool_size
        self.conv_size = conv_size

        self.conv1 = nn.Conv2d(1,conv1_channels,self.conv_size)
        self.conv2 = nn.Conv2d(conv1_channels,conv2_channels,self.conv_size)
        self.conv3 = nn.Conv2d(conv2_channels,conv3_channels,self.conv_size)

        self.number_features = num_features
        self.fc = nn.Linear(in_features=self.number_features, out_features=m)

    def forward(self,audio):
        
        audio = audio[:,None,:,:]

        conv_output1 = self.conv1(audio)
        output1 = F.relu(nn.MaxPool2d(self.pool_size)(conv_output1))
        
        conv_output2 = self.conv2(output1)
        output2 = F.relu(nn.MaxPool2d(self.pool_size)(conv_output2))
        
        conv_output3 = self.conv3(output2)
        output3 = F.relu(nn.MaxPool2d(self.pool_size)(conv_output3))

        flattened = output3.view(-1)

        return self.fc(flattened)

class Baseline(torch.nn.Module):
    def __init__(self, avg=.9998,stride=512,regions=25,d=4096,k=500,m=128,stft=512,window=16384,batch_size=100,basechannel=16):
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


class ComplexModel(torch.nn.Module):
    def __init__(self, avg=.9998,stride=512,regions=25,d=4096,k=500,m=128,stft=512,window=16384,batch_size=100,basechannel=16):
        super(ComplexModel, self).__init__()

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
        self.inshape=501*2*basechannel
        self.conv1 = nn.Conv2d(2,basechannel,(128,1),padding=(64,0))
        self.conv2 = nn.Conv2d(basechannel,2*basechannel,(1,25))
        self.linear = nn.Linear(self.inshape,m)
        
        self.norm1 = nn.BatchNorm2d(basechannel)
        self.norm2 = nn.BatchNorm2d(basechannel*2)
    def forward(self, x):
        zx = torch.stack([conv1d(x[:,None,:], self.wsin_var, stride=self.stride),conv1d(x[:,None,:], self.wcos_var, stride=self.stride)],dim=1)
        #batch size * 500 * 25
        x = F.relu(self.conv1(zx))
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



class CrossStitchModel(torch.nn.Module):
    def __init__(self,model_n,model_i):
        super(CrossStitchModel,self).__init__
        self.model_n = model_n
        self.model_i = model_i
        self.cross_matrix = Variable(torch.tensor([2,2]).random_(from =0.1,to =0.9))
    def forward(self,x):
        zx = conv1d(x[:,None,:], self.wsin_var, stride=self.stride).pow(2) \
           + conv1d(x[:,None,:], self.wcos_var, stride=self.stride).pow(2)
                   #batch size * 500 * 25
        zx = zx.unsqueeze(1)
        
        x1 = F.relu(self.model_n.conv1(torch.log(zx + 10e-15)))
        x2 = F.relu(self.model_i.conv1(torch.log(zx + 10e-15)))
        

        # batch size *basechannel * 501 * 25
        x1 = self.model_n.norm1(x1)
        x1 =  F.relu(self.model_n.conv2(x1))

        x2 = self.model_i.norm1(x2)
        x2 =  F.relu(self.model_i.conv2(x2))


        # Crossing
        x1 = self.cross_matrix[0,0]*x1 + self.cross_matrix[1,0]*x2
        x2 = self.cross_matrix[0,1]*x1 + self.cross_matrix[1,1]*x2
        

        # batchsize * basechannel2 * 501 * 1
        x1 = self.model_n.norm2(x1)
        x1 = x1.reshape(self.model_n.batch_size,self.model_n.inshape)

        # batch size *basechannel * 501 * 25

        # batchsize * basechannel2 * 502 * 2
        x2 = self.model_i.norm2(x2)
        x2 = x2.reshape(self.model_i.batch_size,self.model_i.inshape)
        return torch.cat((self.model_n.linear(x1)), self.model_i.linear(x2),dim=1)
