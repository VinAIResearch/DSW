import numpy as np
import torch 
from torch import optim
from mlp import MLP
from utils import *
#https://github.com/kimiandj/gsw
class GSW_NN():
    def __init__(self,din=2,nofprojections=10,model_depth=3,num_filters=32,use_cuda=True):        

        self.nofprojections=nofprojections

        if torch.cuda.is_available() and use_cuda:
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')
        
        self.parameters=None # This is for max-GSW
        self.din=din
        self.dout=nofprojections
        self.model_depth=model_depth
        self.num_filters=num_filters
        self.model=MLP(din=self.din,dout=self.dout,num_filters=self.num_filters)
        if torch.cuda.is_available() and use_cuda:
            self.model.cuda()
 
    def gsw(self,X,Y,random=True):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        '''
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N
        
        if random:
            self.model.reset()
        
        Xslices=self.model(X.to(self.device))
        Yslices=self.model(Y.to(self.device))

        Xslices_sorted=torch.sort(Xslices,dim=0)[0]
        Yslices_sorted=torch.sort(Yslices,dim=0)[0]
        
        return torch.sqrt(torch.sum((Xslices_sorted-Yslices_sorted)**2))
    def dgsw(self,X,Y,iterations=50,lam=1,lr=1e-4):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        '''
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N
        
        self.model.reset()
        optimizer=optim.Adam(self.model.parameters(),lr=lr)
        for i in range(iterations):
            optimizer.zero_grad()
            Xslices=self.model(X.to(self.device))
            Yslices=self.model(Y.to(self.device))

            Xslices_sorted=torch.sort(Xslices,dim=0)[0]
            Yslices_sorted=torch.sort(Yslices,dim=0)[0]
            loss = - torch.sqrt(torch.sum((Xslices_sorted-Yslices_sorted)**2)) + lam*cosine_distance_torch(Xslices,Yslices)
            loss.backward(retain_graph=True)
            optimizer.step()
        return self.gsw(X.to(self.device),Y.to(self.device),random=False)

    def max_gsw(self,X,Y,iterations=50,lr=1e-4):
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N

        self.model.reset()
        
        optimizer=optim.Adam(self.model.parameters(),lr=lr)
        total_loss=np.zeros((iterations,))
        for i in range(iterations):
            optimizer.zero_grad()
            loss=-self.gsw(X.to(self.device),Y.to(self.device),random=False)
            total_loss[i]=loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
        
        return self.gsw(X.to(self.device),Y.to(self.device),random=False)
