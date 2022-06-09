# python test_wae.py --source-data MNIST
# python test_wae.py --source-data CelebA --landmark-interval 1
import argparse
import pickle
import torch.utils.data
import Methods.wae as wae
import Methods.evaluation as evaluation
from Methods.models import load_datasets, AE_CelebA, AE_MNIST
import numpy as np
import random
import os

parser = argparse.ArgumentParser(description='Wasserstein Autoencoder Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=2022, metavar='S',
                    help='random seed (default: 2022)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--source-data', type=str, default='MNIST',
                    help='data name')
parser.add_argument('--datapath', type=str, default='Data',
                    help='data path')
parser.add_argument('--resultpath', type=str, default='Results',
                    help='result path')
parser.add_argument('--landmark-interval', type=int, default=10,
                    help='interval for recording')
parser.add_argument('--gamma', type=float, default=100.0,
                    help='the weight of regularizer')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--loss-type', type=str, default='MSE',
                    help='the type of loss')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
# set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if args.cuda else "cpu")
print(device)

if not os.path.isdir(args.resultpath):
    os.mkdir(args.resultpath)


if __name__ == '__main__':
    if args.source_data == 'MNIST':
        args.x_dim = int(28 * 28)
        args.z_dim = 8
        args.nc = 1
        model = AE_MNIST(z_dim=args.z_dim, nc=args.nc)
    else:
        args.x_dim = int(64 * 64)
        args.z_dim = 64
        args.nc = 3
        model = AE_CelebA(z_dim=args.z_dim, nc=args.nc)

    src_loaders = load_datasets(args=args)
    loss = wae.train_model(model, src_loaders['train'], src_loaders['val'], device, args)

    # conditional generation
    model.eval()
    evaluation.sampling(model, device, args.epochs, args, prefix='wae', nrow=4)

    # save models and learning results
    model = model.to('cpu')
    torch.save(model.state_dict(),
               '{}/wae_model_{}.pt'.format(args.resultpath, args.source_data))
    with open('{}/wae_loss_{}.pkl'.format(args.resultpath, args.source_data), 'wb') as f:
        pickle.dump(loss, f)
    print('\n')
