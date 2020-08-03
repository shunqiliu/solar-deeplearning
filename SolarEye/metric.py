import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torch.nn import functional as F
import os

def class_error(net,datal):
  loss=0
  for j,data in enumerate(datal):
    i,p,f=data['image'].float().cuda(),data['pvpl'].float().cuda(),data['env'].float().cuda()
    out=net(i,f)
    _,out=torch.max(out.data,1)
    p=(p*100//12.5).long()
    for e in range(out.shape[0]):
        if out[e]!=p[e]:
            loss+=1
    loss=loss/len(out)
  return loss/(j+1)

def qr_loss(net,datal):
  loss=0
  for j,data in enumerate(datal):
    i,p,f=data['image'].float().cuda(),data['pvpl'].float().cuda(),data['env'].float().cuda()
    
    out=net(i,f)
    qg=F.softmax(out,dim=1)
    for i in range(qg.shape[0]):
      for ii in range(qg.shape[1]):
          if ii==0:
              qg[i,ii]=qg[i,ii]
          else:
              qg[i,ii]+=qg[i,ii-1]
    
    qg.unsqueeze(0).cuda()
    y_true=torch.ones(p.shape[0],8).cuda()*p.view(p.shape[0],1)
    y_pred=torch.ones(p.shape[0],8).cuda()*torch.FloatTensor([12.5,25.0,37.5,50.0,62.5,75.0,87.5,100.0]).cuda()
    e=(y_true-y_pred).unsqueeze(0)
    temp=torch.cat((qg*e,(qg-1)*e))
    temp=torch.max(temp,dim=0)[0]
    loss+=torch.mean(temp)
  return loss/(j+1)

def pin_ball_loss(y_true,y_pred):
  qg=(torch.FloatTensor(range(1,100))/100.0).unsqueeze(0).cuda()
  e=(y_true-y_pred).unsqueeze(0)
  temp=torch.cat((qg*e,(qg-1)*e))
  temp=torch.max(temp,dim=0)[0]
  loss=torch.mean(temp)
  return loss

def compute_test_loss(net,test):
  loss=0
  for j,data in (enumerate(test)):
    i,p,f=data['image'].float().cuda(),data['pvpl'].float().cuda(),data['env'].float().cuda()
    p=p.view(-1,1)
    pnew=torch.ones((p.shape[0],99),dtype=float).cuda()
    p=p*pnew*100
    out=net(i,f)
    loss+=pin_ball_loss(p,out)
  return loss/(j+1)
