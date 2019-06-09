import torch.nn as nn
import torch.functional as F
class DCNN(nn.Module):
    def __init__(self):
        super(DCNN, self).__init__()
    self.conv2048 = nn.conv1d(1,64,2048,strides=512,padding=2048//2)
    self.conv4096 = nn.conv1d(1,64,4096,strides=512,padding=4096//2)
    self.conv8192 = nn.conv1d(1,64,8192,strides=512,padding=8192//2)
    self.conv16384 = nn.conv1d(1,64,16384,strides=512,padding=16384//2)
    def forward(self,x):
        # Size len(input)//512
        c2048 = self.conv2048(x)