import time
import torch
from torch import optim
import Methods.evaluation as evaluation
import Methods.models as method
import numpy as np
import sys
sys.path.append("../src")
import base
from sklearn.decomposition import PCA
import scipy as sp



def PRHCP(X, Y, q=2, niter=5):

    n, d = X.shape
    t = 0
    tau = 1
    i = 0
    Omega = np.eye(d)

    while i<niter:
        if i==0:
            Xr = base.hilbert_order(X.T)
            Yr = base.hilbert_order(Y.T)
            delta = X[Xr,:]-Y[Yr,:]
        else:
            Xu = X@U
            Yu = Y@U
            Xr = base.hilbert_order(Xu.T)
            Yr = base.hilbert_order(Yu.T)
            delta = X[Xr,:]-Y[Yr,:]
        
        disp = np.concatenate([delta, -delta])    
        pca = PCA(n_components = q, random_state = 1)
        pca.fit(disp)
        U = pca.components_.T
        Omega = (1-tau)*Omega + tau*U@U.T
        eigenvalues, eigenvectors = sp.linalg.eigh(Omega, eigvals=(d-q,d-1))
        U = eigenvectors
        t += 1
        tau = 2/(2+t)
        i += 1
    
    return U


# PRHCP-AE
def prhcp_distance(encoded_samples, p=2, device='cpu'):
    
    distribution_samples = torch.randn(size=encoded_samples.size()).to(device)

    # generate max directions
    X = encoded_samples.detach().cpu().numpy()
    Y = distribution_samples.detach().cpu().numpy()
    proj = PRHCP(X, Y, q=2, niter=5)
    projections = torch.from_numpy(proj).type(torch.FloatTensor).to(device)

    # project samples
    encoded_projections = encoded_samples.matmul(projections)
    distribution_projections = distribution_samples.matmul(projections)


    A = base.hilbert_order((X@proj).T)
    B = base.hilbert_order((Y@proj).T)

    hilbert_distance = encoded_projections[A,:] - distribution_projections[B,:]
    hilbert_distance = torch.pow(hilbert_distance, p)
    wd = hilbert_distance.sum()

    return wd


def train(model, train_loader, optimizer, device, epoch, args):
    model.train()
    train_rec_loss = 0
    train_reg_loss = 0
    epoch_time = time.time()
    for batch_idx, (data, _) in enumerate(train_loader):
        since = time.time()
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, z = model(data)
        rec_loss = method.loss_function(recon_batch, data, args.loss_type)
        reg_loss = args.gamma*prhcp_distance(z, device=device)
        loss = rec_loss + reg_loss
        loss.backward()
        train_rec_loss += rec_loss.item()
        train_reg_loss += reg_loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
            print('Time = {:.2f}sec'.format(time.time() - since))

    print('====> Epoch: {} Average RecLoss: {:.4f} RegLoss: {:.4f} TotalLoss: {:.4f}'.format(
        epoch, train_rec_loss / len(train_loader.dataset), train_reg_loss / len(train_loader.dataset),
        (train_rec_loss + train_reg_loss) / len(train_loader.dataset)))

    print('Epoch Time = {:.2f}sec'.format(time.time() - epoch_time))


def test(model, test_loader, device, args):
    model.eval()
    test_rec_loss = 0
    test_reg_loss = 0
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, z = model(data)
            rec_loss = method.loss_function(recon_batch, data, args.loss_type)
            reg_loss = args.gamma * prhcp_distance(z, device=device)
            test_rec_loss += rec_loss.item()
            test_reg_loss += reg_loss.item()
            test_loss += (rec_loss.item() + reg_loss.item())

    test_rec_loss /= len(test_loader.dataset)
    test_reg_loss /= len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('====> Test set RecLoss: {:.4f} RegLoss: {:.4f} TotalLoss: {:.4f}'.format(
        test_rec_loss, test_reg_loss, test_loss))
    return test_rec_loss, test_reg_loss, test_loss


def train_model(model, train_loader, test_loader, device, args):
    model = model.to(device)
    loss_list = []
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, device, epoch, args)
        test_rec_loss, test_reg_loss, test_loss = test(model, test_loader, device, args)
        loss_list.append([test_rec_loss, test_reg_loss, test_loss])
        if epoch % args.landmark_interval == 0:
            evaluation.interpolation_2d(model, test_loader, device, epoch, args, prefix='prhcp-ae')
            evaluation.sampling(model, device, epoch, args, prefix='prhcp-ae')
            evaluation.reconstruction(model, test_loader, device, epoch, args, prefix='prhcp-ae')
    return loss_list
