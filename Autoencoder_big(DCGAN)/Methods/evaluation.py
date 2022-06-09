import numpy as np
import torch
from torchvision.utils import save_image


def interpolation_2d(model, data_loader, device, epoch, args, prefix, nrow=14):
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            if i == 0:
                data = data.to(device)
                recon_batch, mu = model(data)
                mu = mu[:4, :]
            else:
                break

        latents = torch.randn(int(nrow ** 2), mu.size(1)).to(device)
        for i in range(nrow):
            for j in range(nrow):
                x1 = (nrow - 1 - i) / (nrow - 1)
                x2 = i / (nrow - 1)
                y1 = (nrow - 1 - j) / (nrow - 1)
                y2 = j / (nrow - 1)
                n = nrow * i + j
                latents[n, :] = y1 * (x1 * mu[0, :] + x2 * mu[1, :]) + y2 * (x1 * mu[2, :] + x2 * mu[3, :])

        samples = model.decode(latents).cpu()
        s = int(args.x_dim ** 0.5)
        save_image(samples.view(int(nrow ** 2), args.nc, s, s),
                   '{}/{}_interp2d_{}_{}.png'.format(
                       args.resultpath, prefix, args.source_data, epoch), nrow=nrow)


def sampling(model, device, epoch, args, prefix, nrow=14):
    model.eval()
    n_samples = int(nrow ** 2)
    with torch.no_grad():
        sample = torch.randn(n_samples, args.z_dim)
        sample = model.decode(sample.to(device)).cpu()
        s = int(args.x_dim ** 0.5)
        pathname = '{}/{}_samples_{}_{}.png'.format(args.resultpath, prefix, args.source_data, epoch)
        print(pathname)
        save_image(sample.view(n_samples, args.nc, s, s), pathname, nrow=nrow)
        print('Done!')


def reconstruction(model, test_loader, device, epoch, args, prefix, nrow=14):
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu = model(data)
            if i < int(nrow / 2):
                n = min(data.size(0), nrow)
                if i == 0:
                    comparison = torch.cat([data[:n], recon_batch[:n]])
                else:
                    comparison = torch.cat([comparison, data[:n], recon_batch[:n]])
            else:
                break

        save_image(comparison.cpu(),
                   '{}/{}_recon_{}_{}.png'.format(args.resultpath, prefix, args.source_data, epoch), nrow=nrow)



