import numpy as np
import torch
from torch import optim

import sys
sys.path.append("../src")
import base



class GSW():
    def __init__(self,ftype='linear',nofprojections=10,degree=2,radius=2.,lossp=2,use_cuda=True):
        self.ftype=ftype
        self.nofprojections=nofprojections
        self.degree=degree
        self.radius=radius
        self.lossp=lossp
        if torch.cuda.is_available() and use_cuda:
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')
        self.theta=None # This is for max-GSW

    def gsw(self,X,Y,p=2,theta=None):
       
        N,dn = X.shape
        M,dm = Y.shape
        p = self.lossp
        assert dn==dm and M==N

        # add HCP
        if self.ftype=='HCP':

            Xh = X.detach().cpu().numpy()
            Xr = base.hilbert_order(Xh)
            Yh = Y.detach().cpu().numpy()
            Yr = base.hilbert_order(Yh)
         
            return torch.pow(2*torch.mean((X[Xr,:]-Y[Yr,:])**p),1/p)

        else:
            if theta is None:
                theta=self.random_slice(dn)

            Xslices=self.get_slice(X,theta)
            Yslices=self.get_slice(Y,theta)

            Xslices_sorted=torch.sort(Xslices,dim=0)[0]
            Yslices_sorted=torch.sort(Yslices,dim=0)[0]
            return torch.sqrt(torch.mean((Xslices_sorted-Yslices_sorted)**2))

    def max_gsw(self,X,Y,iterations=50,lr=1e-4):
        N,dn = X.shape
        M,dm = Y.shape
        device = self.device
        assert dn==dm and M==N

        if self.ftype=='linear':
            theta=torch.randn((1,dn),device=device,requires_grad=True)
            theta.data/=torch.sqrt(torch.sum((theta.data)**2))
        elif self.ftype=='poly':
            dpoly=self.homopoly(dn,self.degree)
            theta=torch.randn((1,dpoly),device=device,requires_grad=True)
            theta.data/=torch.sqrt(torch.sum((theta.data)**2))
        self.theta=theta

        optimizer=optim.Adam([self.theta],lr=lr)
        total_loss=np.zeros((iterations,))
        for i in range(iterations):
            optimizer.zero_grad()
            loss=-self.gsw(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
            total_loss[i]=loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            self.theta.data/=torch.sqrt(torch.sum(self.theta.data**2))

        return self.gsw(X.to(self.device),Y.to(self.device),self.theta.to(self.device))


    def get_slice(self,X,theta):
       
        if self.ftype=='linear':
            return self.linear(X,theta)
        elif self.ftype=='poly':
            return self.poly(X,theta)
        else:
            raise Exception('Defining function not implemented')


    def random_slice(self,dim):
        if self.ftype=='linear':
            theta=torch.randn((self.nofprojections,dim))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        elif self.ftype=='poly':
            dpoly=self.homopoly(dim,self.degree)
            theta=torch.randn((self.nofprojections,dpoly))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        return theta.to(self.device)


    def linear(self,X,theta):
        if len(theta.shape)==1:
            return torch.matmul(X,theta)
        else:
            return torch.matmul(X,theta.t())


    def poly(self,X,theta):
        N,d=X.shape
        assert theta.shape[1]==self.homopoly(d,self.degree)
        powers=list(self.get_powers(d,self.degree))
        HX=torch.ones((N,len(powers))).to(self.device)
        for k,power in enumerate(powers):
            for i,p in enumerate(power):
                HX[:,k]*=X[:,i]**p
        if len(theta.shape)==1:
            return torch.matmul(HX,theta)
        else:
            return torch.matmul(HX,theta.t())


    def get_powers(self,dim,degree):
        if dim == 1:
            yield (degree,)
        else:
            for value in range(degree + 1):
                for permutation in self.get_powers(dim - 1,degree - value):
                    yield (value,) + permutation


    def homopoly(self,dim,degree):
        return len(list(self.get_powers(dim,degree)))
