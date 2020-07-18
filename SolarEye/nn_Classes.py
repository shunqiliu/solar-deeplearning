import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torch.nn import functional as F
import cv2

#Original network
class SolarSet(Dataset):
    def __init__(self,imgdir,pvpl,env):
        self.images=imgdir
        self.label=pvpl
        self.Env=env

    def __getitem__(self, index):
        img=cv2.imread('../PanelImages/'+self.images[index])
        img=img.transpose(2,0,1)
        item = {'image': img, 'pvpl': self.label[index], 'env': self.Env[index]}
        return item
    
    def __len__(self):
        return len(self.images)

class ImpactNet(nn.Module):
  def __init__(self):
    super().__init__()
    #Define dissembled operators
    self.dropout = nn.Dropout(p=0.5)
    self.relu=nn.ReLU()
    self.conv1=nn.Conv2d(3,16,7,padding=3)
    self.pool=nn.AvgPool2d(3)
    #AU1
    self.rcu1_conv=nn.Conv2d(16,32,1,2)
    self.rcu1=nn.Sequential(
        nn.Conv2d(32,32,5,padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32,32,5,padding=2),
        nn.BatchNorm2d(32)
        )
    #AU2
    self.rcu2_conv=nn.Conv2d(32,48,1,2)
    self.rcu2=nn.Sequential(
        nn.Conv2d(48,48,5,padding=2),
        nn.BatchNorm2d(48),
        nn.ReLU(),
        nn.Conv2d(48,48,5,padding=2),
        nn.BatchNorm2d(48)
        )
    #AU3
    self.rcu3_conv=nn.Conv2d(48,64,1,2)
    self.rcu3=nn.Sequential(
        nn.Conv2d(64,64,5,padding=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64,64,5,padding=2),
        nn.BatchNorm2d(64)
        )
    #AU4
    self.rcu4_conv=nn.Conv2d(64,80,1,2)
    self.rcu4=nn.Sequential(
        nn.Conv2d(80,80,5,padding=2),
        nn.BatchNorm2d(80),
        nn.ReLU(),
        nn.Conv2d(80,80,5,padding=2),
        nn.BatchNorm2d(80)
        )
    #AU5
    self.rcu5_conv=nn.Conv2d(80,96,1,2)
    self.rcu5=nn.Sequential(
        nn.Conv2d(96,96,5,padding=2),
        nn.BatchNorm2d(96),
        nn.ReLU(),
        nn.Conv2d(96,96,5,padding=2),
        nn.BatchNorm2d(96)
        )
    #FCs in Analysis unit part
    self.fu=nn.Linear(384,96)
    self.fc0=nn.Linear(96,96)

    #FC layers
    self.fc1=nn.Linear(4,64)
    self.fc2=nn.Linear(64,96)
    self.fc3=nn.Linear(192,32)
    self.fc4=nn.Linear(32,8)

  def forward(self,au,ef):
    #AU is output from images, EF is output from environment factors

    #Analysis network
    au=self.conv1(au)#1st layer: only conv2d
    au=self.relu(self.rcu1_conv(au)+self.rcu1(self.rcu1_conv(au)))#2nd layer: residual convolution unit
    au=self.relu(self.rcu2_conv(au)+self.rcu2(self.rcu2_conv(au)))#3rd layer: residual convolution unit
    au=self.relu(self.rcu3_conv(au)+self.rcu3(self.rcu3_conv(au)))#4th layer: residual convolution unit
    au=self.relu(self.rcu4_conv(au)+self.rcu4(self.rcu4_conv(au)))#5th layer: residual convolution unit
    au=self.relu(self.rcu5_conv(au)+self.rcu5(self.rcu5_conv(au)))#6th layer: residual convolution unit
    au=self.pool(au)#7th layer: maxpool 2d
    au=au.view(au.shape[0],-1)
    au=self.relu(self.dropout(self.fu(au)))
    au=self.relu(self.dropout(self.fc0(au)))

    #Fully Connect network
    ef=self.relu(self.dropout(self.fc1(ef)))
    ef=self.relu(self.dropout(self.fc2(ef)))
    
    o=torch.cat((ef,au),1)
    o=self.relu(self.dropout(self.fc3(o)))
    o=self.fc4(o)

    return o

class SolarQRNN(nn.Module):
  def __init__(self):
    super().__init__()
    #Define dissembled operators
    self.dropout = nn.Dropout(p=0.5)
    self.relu=nn.ReLU()
    self.conv1=nn.Conv2d(3,16,7,padding=3)
    self.pool=nn.AvgPool2d(3)
    #AU1
    self.rcu1_conv=nn.Conv2d(16,32,1,2)
    self.rcu1=nn.Sequential(
        nn.Conv2d(32,32,5,padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32,32,5,padding=2),
        nn.BatchNorm2d(32)
        )
    #AU2
    self.rcu2_conv=nn.Conv2d(32,48,1,2)
    self.rcu2=nn.Sequential(
        nn.Conv2d(48,48,5,padding=2),
        nn.BatchNorm2d(48),
        nn.ReLU(),
        nn.Conv2d(48,48,5,padding=2),
        nn.BatchNorm2d(48)
        )
    #AU3
    self.rcu3_conv=nn.Conv2d(48,64,1,2)
    self.rcu3=nn.Sequential(
        nn.Conv2d(64,64,5,padding=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64,64,5,padding=2),
        nn.BatchNorm2d(64)
        )
    #AU4
    self.rcu4_conv=nn.Conv2d(64,80,1,2)
    self.rcu4=nn.Sequential(
        nn.Conv2d(80,80,5,padding=2),
        nn.BatchNorm2d(80),
        nn.ReLU(),
        nn.Conv2d(80,80,5,padding=2),
        nn.BatchNorm2d(80)
        )
    #AU5
    self.rcu5_conv=nn.Conv2d(80,96,1,2)
    self.rcu5=nn.Sequential(
        nn.Conv2d(96,96,5,padding=2),
        nn.BatchNorm2d(96),
        nn.ReLU(),
        nn.Conv2d(96,96,5,padding=2),
        nn.BatchNorm2d(96)
        )
    #FCs
    self.fu=nn.Linear(384,96)
    self.fc0=nn.Linear(96,96)

    
    self.fc1=nn.Linear(4,64)
    self.fc2=nn.Linear(64,96)
    self.fc3=nn.Linear(192,32)
    self.fc4=nn.Linear(32,99)

  def forward(self,au,ef):
    #AU is output from images, EF is output from environment factors

    #Analysis network
    au=self.conv1(au)#1st layer: only conv2d
    au=self.relu(self.rcu1_conv(au)+self.rcu1(self.rcu1_conv(au)))#2nd layer: residual convolution unit
    au=self.relu(self.rcu2_conv(au)+self.rcu2(self.rcu2_conv(au)))#3rd layer: residual convolution unit
    au=self.relu(self.rcu3_conv(au)+self.rcu3(self.rcu3_conv(au)))#4th layer: residual convolution unit
    au=self.relu(self.rcu4_conv(au)+self.rcu4(self.rcu4_conv(au)))#5th layer: residual convolution unit
    au=self.relu(self.rcu5_conv(au)+self.rcu5(self.rcu5_conv(au)))#6th layer: residual convolution unit
    au=self.pool(au)#7th layer: maxpool 2d
    au=au.view(au.shape[0],-1)
    au=self.relu(self.dropout(self.fu(au)))
    au=self.relu(self.dropout(self.fc0(au)))

    #Fully Connect network
    ef=self.relu(self.dropout(self.fc1(ef)))
    ef=self.relu(self.dropout(self.fc2(ef)))
    
    o=torch.cat((ef,au),1)
    o=self.relu(self.dropout(self.fc3(o)))
    o=self.fc4(o)
    

    return o