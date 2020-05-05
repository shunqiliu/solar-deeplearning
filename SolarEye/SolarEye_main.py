import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import nn_Classes
import load_data
import metric
import data_pre

def Pin_origin():
    #train Resnet
    epochs=30
    lr=0.00001

    belos=[]
    mselos=[]
    l1los=[]
    qrlos=[]
    net=nn_Classes.ImpactNet_A()
    net.cuda()
    criteria=nn.CrossEntropyLoss()
    for ii in tqdm(range(epochs)):
        if ii%10==0:
            sgd=torch.optim.SGD(net.parameters(),lr)
            lr=lr*0.1
        for j,data in enumerate(train_dataloader):
            i,p,f=data['image'].float().cuda(),data['power'],data['feats'].float().cuda()
            p=(p*100//12.5).long().cuda()
            sgd.zero_grad()
            out=net(i,f)
            loss=criteria(out,p)
            loss.backward()
            sgd.step()
        with torch.no_grad():
            #Pinball loss
            qrlos.append(metric.qrloss(net,test))

            """
            #Experiment for report 1
            belos.append(metric.error(net,test,etype='BE'))
            mselos.append(metric.error(net,test,etype='MSE'))
            l1los.append(metric.error(net,test,etype='L1'))
            """

    #Plot loss curve
    t=list(range(ii+1))

    #Pinball loss
    plt.figure()
    plt.title('Pinball Loss for original network')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.plot(t,qrlos)

    """
    #Experiment for report 1
    plt.figure()
    plt.title('Binary error')
    plt.xlabel('Time')
    plt.ylabel('Error %')
    plt.plot(t,belos)

    plt.figure()
    plt.title('MSE loss')
    plt.xlabel('Time')
    plt.ylabel('Error %')
    plt.plot(t,mselos)

    plt.figure()
    plt.title('L1 loss')
    plt.xlabel('Time')
    plt.ylabel('Error %')
    plt.plot(t,l1los)
    """

    plt.show()

def Pin_modify():
    #train Resnet
    epochs=30
    lr=0.00001

    pinball_loss=[]
    net=nn_Classes.NewNet()
    net.cuda()
    for ii in tqdm(range(epochs)):
        if ii%10==0:
            sgd=torch.optim.SGD(net.parameters(),lr)
            lr=lr*0.1
        net.train()
        for j,data in (enumerate(train_dataloader)):
            i,p,f=data['image'].float().cuda(),data['power'].float().cuda(),data['feats'].float().cuda()
            p=p.view(-1,1)
            pnew=torch.ones((p.shape[0],99),dtype=float).cuda()
            p=p*pnew*100
            sgd.zero_grad()
            out=net(i,f)
            loss=metric.pin_ball_loss(p,out)
            loss.backward()
            sgd.step()
        net.eval()
        with torch.no_grad():
            pinball_loss.append(metric.compute_test_loss(net,test))

    #plot loss curve
    t=list(range(ii+1))
    plt.figure()
    plt.title('Pinball loss for modified network')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.plot(t,pinball_loss)

    plt.show()

if __name__ == '__main__':
    #data_pre.load_origin_pic_label_feat()          #Save the label,environmental features and images to .npy
    assert(torch.cuda.is_available()==True,'Cuda is unavailable, please check your GPU type')
    print("Version:", torch.__version__)

    #Pre-process data (already done, no need to run it)
    #load_origin_pic_label_feat()

    #load data
    train,test=load_data.load_set()
    train_dataloader=DataLoader(train, 32)
    Pin_origin()
    Pin_modify()
