import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class AE_MNIST(nn.Module):
    def __init__(self, z_dim=8, nc=1):
        super(AE_MNIST, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),            
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024 * 1 * 1))   
        )

        self.fc = nn.Linear(1024 * 1 * 1, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024 * 7 * 7),  
            View((-1, 1024, 7, 7)),   
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),   
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, nc, 1)                       
        )

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def encode(self, x):
        z = self.encoder(x)
        z = self.fc(z)
        return z

    def decode(self, z):
        return self.decoder(z)


class AE_CelebA(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super(AE_CelebA, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),             
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024 * 4 * 4))         
        )

        self.fc = nn.Linear(1024 * 4 * 4, z_dim)    

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024 * 8 * 8),   
            View((-1, 1024, 8, 8)),  
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),                       
        )

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def encode(self, x):
        z = self.encoder(x)
        z = self.fc(z)
        return z

    def decode(self, z):
        return self.decoder(z)


def load_datasets(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    name = args.source_data
    pathname = args.datapath
    data_set = getattr(datasets, name)
    data_loaders = {}
    print(name)
    if name == 'MNIST':
        data_loaders['train'] = torch.utils.data.DataLoader(
            data_set(pathname, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        data_loaders['val'] = torch.utils.data.DataLoader(
            data_set(pathname, train=False, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif name == 'CelebA':
        transform = transforms.Compose([
            transforms.CenterCrop((140, 140)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ])
        data_loaders['train'] = torch.utils.data.DataLoader(
            data_set(pathname, split='train', download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        data_loaders['val'] = torch.utils.data.DataLoader(
            data_set(pathname, split='test', download=True, transform=transform),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        print('Unknown data class!')

    return data_loaders


def loss_function(recon_x, x, rec_type):
    # if rec_type == 'BCE':
    #     reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    if rec_type == 'MSE':
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')
    elif rec_type == 'MAE':
        reconstruction_loss = F.l1_loss(recon_x, x, reduction='sum')
    else:
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum') + F.l1_loss(recon_x, x, reduction='sum')
    return reconstruction_loss

