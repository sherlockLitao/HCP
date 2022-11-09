import time
import torch
from torch import optim
import Methods.evaluation as evaluation
import Methods.models as method
import numpy as np
from scipy.stats import special_ortho_group
import sys
sys.path.append("../src")
import base


def rand_projections(embedding_dim, num_samples=50):

    projections = [w / np.sqrt((w**2).sum())
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor)


# IPRHCP-AE
def iprhcp_distance(encoded_samples, num_projections=50, p=2, device='cpu'):
    
    embedding_dim = encoded_samples.size(1)
    distribution_samples = torch.randn(size=encoded_samples.size()).to(device)

    # generate orthogonal directions
    nm = np.int32(num_projections*2/embedding_dim+1)
    projections = np.zeros((nm*embedding_dim,embedding_dim))
    for l in range(nm):
        projections[(embedding_dim*l):(embedding_dim*(l+1)),:] = special_ortho_group.rvs(embedding_dim)
    
    projections = torch.from_numpy(projections).type(torch.FloatTensor).to(device)

    # project samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))


    A = encoded_projections.detach().cpu().numpy()
    B = distribution_projections.detach().cpu().numpy()

    # start: i=0
    i = 0
    A1 = base.hilbert_order(A[:,(2*i):(2*i+2)])
    B1 = base.hilbert_order(B[:,(2*i):(2*i+2)])

    hilbert_distance = encoded_projections[A1,(2*i):(2*i+2)] - distribution_projections[B1,(2*i):(2*i+2)]
    hilbert_distance = torch.pow(hilbert_distance, p)
    wd = hilbert_distance.sum()

    # for i>0
    for i in range(1,num_projections):
    
        A1 = base.hilbert_order(A[:,(2*i):(2*i+2)])
        B1 = base.hilbert_order(B[:,(2*i):(2*i+2)])

        hilbert_distance = encoded_projections[A1,(2*i):(2*i+2)] - distribution_projections[B1,(2*i):(2*i+2)]
        hilbert_distance = torch.pow(hilbert_distance, p)
        hilbert_distance = hilbert_distance.sum()
        wd = wd + hilbert_distance

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
        reg_loss = args.gamma*iprhcp_distance(z, device=device)
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
            reg_loss = args.gamma * iprhcp_distance(z, device=device)
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
            evaluation.interpolation_2d(model, test_loader, device, epoch, args, prefix='iprhcp-ae')
            evaluation.sampling(model, device, epoch, args, prefix='iprhcp-ae')
            evaluation.reconstruction(model, test_loader, device, epoch, args, prefix='iprhcp-ae')
    return loss_list
