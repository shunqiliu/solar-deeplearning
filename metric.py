import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torch.nn import functional as F

def error(net,test,etype='L1'):
    """
    Metrics for experiment 1
    Input:
        net:trained neaural network
        test:test data
        etype:'L1' for L1-loss, 'MSE' for Mean Square Error, 'BE' for classification error
    Return:
        mean of corresponding loss
    """
    j=0
    loss=0
    for data in test:
        j+=1
        i,p,f=torch.FloatTensor(data['image']).view(-1,3,192,192).cuda(),torch.FloatTensor([data['power']]).cuda(),torch.FloatTensor(data['feats']).view(1,-1).cuda()
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
    """
    Metrics for Pinball loss on original network. Instruction can be found in README.md
    Input:
        net:trained neaural network
        test:test data
    Return:
        Pinball loss
    """
    j=0
    loss=0
    for data in test:
        j+=1
        i,p,f=torch.FloatTensor(data['image']).view(-1,3,192,192).cuda(),torch.FloatTensor([data['power']]),torch.FloatTensor(data['feats']).view(1,-1).cuda()
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
    
def pin_ball_loss(y_true,y_pred):
    """
    Metrics for Pinball loss on modified network
    Input:
        y_true:ground truth
        y_pred:predicion
    Return:
        Pinball loss
    """
    qg=(torch.FloatTensor(range(1,100))/100.0).unsqueeze(0).cuda()
    e=(y_true-y_pred).unsqueeze(0)
    temp=torch.cat((qg*e,(qg-1)*e))
    temp=torch.max(temp,dim=0)[0]
    loss=torch.mean(temp)
    return loss

def compute_test_loss(net,test):
    """
    Compute Pinball loss in test data
    Input:
        net:trained neaural network
        test:test data
    Return:
        Pinball loss for test data
    """
    j=0
    loss=0
    for data in test:
        j+=1
        i,p,f=torch.FloatTensor(data['image']).view(-1,3,192,192).cuda(),torch.FloatTensor([data['power']]).cuda(),torch.FloatTensor(data['feats']).view(1,-1).cuda()
        out=net(i,f)
        loss+=pin_ball_loss(p,out)
    return loss/j