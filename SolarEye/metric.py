import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torch.nn import functional as F
import os

def error(net,test,etype='L1'):
  j=0
  loss=0
  for data in test:
    j+=1
    i,p,f=torch.FloatTensor(data['image']).view(-1,3,192,192).cuda(),torch.FloatTensor([data['pvpl']]).cuda(),torch.FloatTensor(data['env']).view(1,-1).cuda()
    out=net(i,f)
    _,out=torch.max(out.data,1)
    if etype=='L1':
      loss+=torch.abs((out*12.5+6.25)-p)
    elif etype=='MSE':
      loss+=((out*12.5+6.25)-p)**2
    elif etype=='BE':
      p=(p*100//12.5).long()
      if p!=out:
        loss+=1
  return loss/j

def qrloss(net,test):
  j=0
  loss=0
  for data in test:
    j+=1
    i,p,f=torch.FloatTensor(data['image']).view(-1,3,192,192).cuda(),torch.FloatTensor([data['pvpl']]),torch.FloatTensor(data['env']).view(1,-1).cuda()
    out=net(i,f).squeeze().cpu()
    qg=torch.sort(F.softmax(out,dim=0))[0]
    s=0
    for i in range(len(qg)):
      s+=qg[i]
      qg[i]=s
    qg=qg.unsqueeze(0)
    y_true=(torch.ones(8)*p)
    y_pred=torch.FloatTensor([12.5,25.0,37.5,50.0,62.5,75.0,87.5,100.0])
    e=(y_true-y_pred)
    temp=torch.cat((qg*e,(qg-1)*e))
    temp=torch.max(temp,dim=0)[0]
    loss+=torch.mean(temp)
  return loss/j

def train_loss(net,train):
  j=0
  loss=0
  im,p,f=train['image'].float().cuda(),train['pvpl'],train['env'].float().cuda()
  for ind in range(im.shape[0]):
    j+=1
    image=im[ind].view(-1,3,192,192).cuda()
    factor=f[ind].view(1,-1).cuda()
    power=p[ind]
    out=net(image,factor).squeeze().cpu()
    qg=torch.sort(F.softmax(out,dim=0))[0]
    s=0
    for i in range(len(qg)):
      s+=qg[i]
      qg[i]=s
    qg=qg.unsqueeze(0)
    y_true=(torch.ones(8)*power)
    y_pred=torch.FloatTensor([12.5,25.0,37.5,50.0,62.5,75.0,87.5,100.0])
    e=(y_true-y_pred)
    temp=torch.cat((qg*e,(qg-1)*e))
    temp=torch.max(temp,dim=0)[0]
    loss+=torch.mean(temp)
  return loss/j

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
