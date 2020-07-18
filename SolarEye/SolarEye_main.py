import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from nn_Classes import *
from metric import *
import os

def data_pre():
    #Data prepare
    print("\n Data prepare")
    pim=os.listdir("../PanelImages/")
    pvpl=[]
    enF=[]
    for name in pim:
        str=name.replace('.jpg','').split('_')
        s=[float(str[4]),float(str[6]),float(str[8]),float(str[-1]),float(str[-3])]
        pvpl.append(s[-1])
        enF.append(s[:4])
    print("\tfinished")
    return np.array(pim),np.array(pvpl),np.array(enF)

def data_load(imgdir,pvpl,Env):
    #load data
    print("\n load data")
    index=np.array(list(range(45754)))
    np.random.shuffle(index)
    indr=index[:27537]
    indd=index[27537:36645]
    indt=index[36645:]

    train=SolarSet(imgdir[indr],pvpl[indr],Env[indr])
    test=SolarSet(imgdir[indt],pvpl[indt],Env[indt])
    dev=SolarSet(imgdir[indd],pvpl[indd],Env[indd])

    train_loader=DataLoader(train, 64,shuffle=True,num_workers=5)
    test_loader=DataLoader(test, 64,shuffle=True,num_workers=5)
    dev_loader=DataLoader(dev, 64,shuffle=True,num_workers=5)
    print("\tfinished")
    return train_loader,test_loader,dev_loader

def train_impactnet(train,test,dev):
    return

def train_SolarQRNN(train,test,dev):
    #train SolarQRNN
    print("\n train SolarQRNN")
    epochs=90
    lr=0.001

    pinball_loss=[]
    train_loss=[]

    Nnet=SolarQRNN()
    Nnet.cuda()
    for ii in tqdm(range(epochs)):
      if ii%30==0:
        sgd=torch.optim.SGD(Nnet.parameters(),lr)
        lr=lr*0.1
      Nnet.train()
      print("\n train")
      for j,data in (enumerate(train)):
        i,p,f=data['image'].float().cuda(),data['pvpl'].float().cuda(),data['env'].float().cuda()
        p=p.view(-1,1)
        pnew=torch.ones((p.shape[0],99),dtype=float).cuda()
        p=p*pnew*100
        sgd.zero_grad()
        out=Nnet(i,f)
        loss=pin_ball_loss(p,out)
        loss.backward()
        sgd.step()
      Nnet.eval()
      print("\n test")
      with torch.no_grad():
        a=compute_test_loss(Nnet,test)
        pinball_loss.append(float(a))
        train_loss.append(float(loss))
        print("trainloss:",float(loss),"\n","testloss:",float(a),"\n")
    torch.save(Nnet.state_dict(), "SolarQRNN.pth")
    print("\tfinished")
    return train_loss,pinball_loss

def plot(title,xlabel,ylabel,trainloss,testloss):
    t=list(range(len(trainloss)))
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(t,trainloss,c='red')
    plt.plot(t,testloss,c='blue')
    
    plt.legend(["trainloss","testloss"])

    plt.show()

def predict_SolarQRNN(test):
    print("\nPredict")
    net=SolarQRNN()
    net.cuda()
    net.load_state_dict(torch.load("SolarQRNN.pth"))
    with torch.no_grad():
        for j,data in (enumerate(test)):
            i,p,f=data['image'].float().cuda(),data['pvpl'].float().cuda(),data['env'].float().cuda()
            p=p.view(-1,1)
            pnew=torch.ones((p.shape[0],99),dtype=float).cuda()
            p=p*pnew*100
            out=net(i,f)
            print(out[10],p[10])
            os.system("Pause")
    print("\tfinished")
    

if __name__ == '__main__':
    print("Version:", torch.__version__)
    imgdir,pvpl,Env=data_pre()
    train,test,dev=data_load(imgdir,pvpl,Env)
    #trainloss.testloss=train_SolarQRNN(train,test,dev)
    #plot("","","",trainloss,testloss)
    predict_SolarQRNN(test)

