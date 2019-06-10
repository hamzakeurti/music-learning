import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import conv1d, mse_loss
import musicnet
import numpy as np
import copy
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


class Baseline(torch.nn.Module):
    def __init__(self, avg=.9998,stride=512,regions=25,d=4096,k=500,m=128):
        super(Baseline, self).__init__()
        
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

class Model(torch.nn.Module):
    def __init__(self, avg=.9998,stride=512,regions=25,d=4096,k=500,m=128,stft=512,window=16384,batch_size=100,basechannel=16):
        super(Model, self).__init__()

        self.stride=stride
        self.regions=regions
        self.k=k

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
        self.conv5 = nn.Conv2d(basechannel*8,basechannel*8,3,padding=1)
        self.inshape = 336*basechannel
        self.linear = nn.Linear(336*basechannel,m)
    def forward(self, x):
        fft = torch.stft(x,self.stft)
        # batch_size * N * T * 2
        afftpow2 = fft[:,:,:,0] **2 + fft[:,:,:,1]
        x = afftpow2.unsqueeze(1)
        # batch_size * N * T
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x =  F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.avgpool(x)
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
