from __future__ import print_function
from MnistAutoencoder import MnistAutoencoder
from TransformNet import TransformNet
import torch
import argparse
import os
from torch import optim
from torchvision import transforms
from experiments import sampling,reconstruct
from tqdm import tqdm
import torchvision.datasets as datasets
from utils import circular_function

# torch.backends.cudnn.enabled = False


def main():
    # train args
    parser = argparse.ArgumentParser(description='Disributional Sliced Wasserstein Autoencoder')
    parser.add_argument('--datadir', default='./', help='path to dataset')
    parser.add_argument('--outdir', default='./result',
                        help='directory to output images')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--num-workers', type=int, default=16, metavar='N',
                        help='number of dataloader workers if device is CPU (default: 16)')
    parser.add_argument('--seed', type=int, default=16, metavar='S',
                        help='random seed (default: 16)')
    parser.add_argument('--g', type=str, default='circular',
                        help='g')
    parser.add_argument('--num-projection', type=int, default=1000,
                        help='number projection')
    parser.add_argument('--lam', type=float, default=1,
                        help='Regularization strength')
    parser.add_argument('--p', type=int, default=2,
                        help='Norm p')
    parser.add_argument('--niter', type=int, default=10,
                        help='number of iterations')
    parser.add_argument('--r', type=float, default=1000,
                        help='R')
    parser.add_argument('--latent-size', type=int, default=32,
                        help='Latent size')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='(MNIST|FMNIST)')
    parser.add_argument('--model-type', type=str, required=True,
                        help='(SWD|MSWD|DSWD|GSWD|DGSWD|JSWD|JMSWD|JDSWD|JGSWD|JDGSWD|CRAMER|JCRAMER|SINKHORN|JSINKHORN)')
    args = parser.parse_args()

    torch.random.manual_seed(args.seed)
    if (args.g == 'circular'):
        g_function = circular_function
    model_type = args.model_type
    latent_size = args.latent_size
    num_projection = args.num_projection
    dataset=args.dataset
    model_dir = os.path.join(args.outdir, model_type)
    assert dataset in ['MNIST', 'FMNIST']
    assert model_type in ['SWD','MSWD','DSWD','GSWD','DGSWD','JSWD','JMSWD','JDSWD','JGSWD','JDGSWD','CRAMER','JCRAMER','SINKHORN','JSINKHORN']
    if not (os.path.isdir(args.datadir)):
        os.makedirs(args.datadir)
    if not (os.path.isdir(args.outdir)):
        os.makedirs(args.outdir)
    if not (os.path.isdir(args.outdir)):
        os.makedirs(args.outdir)
    if not (os.path.isdir(model_dir)):
        os.makedirs(model_dir)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('batch size {}\nepochs {}\nAdam lr {} \n using device {}\n'.format(
        args.batch_size, args.epochs, args.lr, device.type
    ))
    # build train and test set data loaders
    if(dataset=='MNIST'):
        image_size = 28
        num_chanel = 1
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.datadir, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.datadir, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
            batch_size=64, shuffle=False, num_workers=args.num_workers)
        model = MnistAutoencoder(image_size=28, latent_size=args.latent_size, hidden_size=100, device=device).to(device)
    elif(dataset=='FMNIST'):
        image_size = 28
        num_chanel = 1
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(args.datadir, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(args.datadir, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
            batch_size=64, shuffle=False, num_workers=args.num_workers)
        model = MnistAutoencoder(image_size=28, latent_size=args.latent_size, hidden_size=100, device=device).to(device)
    if (model_type == 'DSWD'  or model_type == 'DGSWD'):
        transform_net = TransformNet(28 * 28).to(device)
        op_trannet = optim.Adam(transform_net.parameters(), lr=args.lr, betas=(0.5, 0.999))
        # train_net(28 * 28, 1000, transform_net, op_trannet)
    elif (model_type == 'JDSWD' or model_type == 'JDSWD2' or model_type == 'JDGSWD'):
        transform_net = TransformNet(args.latent_size + 28 * 28).to(device)
        op_trannet = optim.Adam(transform_net.parameters(), lr=args.lr, betas=(0.5, 0.999))
        # train_net(args.latent_size + 28 * 28, 1000, transform_net, op_trannet)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    fixednoise = torch.randn((64, latent_size)).to(device)
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_idx, (data, y) in tqdm(enumerate(train_loader, start=0)):
            if (model_type == 'SWD'):
                loss = model.compute_loss_SWD(data, torch.randn, num_projection, p=args.p)
            elif (model_type == 'GSWD'):
                loss = model.compute_loss_GSWD(data, torch.randn, g_function, args.r, num_projection, p=args.p)
            elif (model_type == 'MSWD'):
                loss, v = model.compute_loss_MSWD(data, torch.randn, p=args.p, max_iter=args.niter)
            elif (model_type == 'DSWD'):
                loss = model.compute_lossDSWD(data, torch.randn, num_projection, transform_net, op_trannet, p=args.p,
                                              max_iter=args.niter, lam=args.lam)
            elif (model_type == 'DGSWD'):
                loss = model.compute_lossDGSWD(data, torch.randn, num_projection, transform_net, op_trannet,
                                               g_function, r=args.r, p=args.p, max_iter=args.niter, lam=args.lam)
            elif (model_type == 'JSWD'):
                loss = model.compute_loss_JSWD(data, torch.randn, num_projection, p=args.p)
            elif (model_type == 'JGSWD'):
                loss = model.compute_loss_JGSWD(data, torch.randn, g_function, args.r, num_projection, p=args.p)
            elif (model_type == 'JDSWD'):
                loss = model.compute_lossJDSWD(data, torch.randn, num_projection, transform_net, op_trannet, p=args.p,
                                               max_iter=args.niter, lam=args.lam)
            elif (model_type == 'JDGSWD'):
                loss = model.compute_lossJDGSWD(data, torch.randn, num_projection, transform_net, op_trannet,
                                                g_function, r=args.r, p=args.p,
                                                max_iter=args.niter, lam=args.lam)
            elif (model_type == 'JMSWD'):
                loss, v = model.compute_loss_MSWD(data, torch.randn, p=args.p, max_iter=args.niter)
            elif (model_type == 'CRAMER'):
                loss = model.compute_loss_cramer(data, torch.randn)
            elif (model_type == 'JCRAMER'):
                loss = model.compute_loss_join_cramer(data, torch.randn)
            elif (model_type == 'WVI'):
                loss = model.compute_wasserstein_vi_loss(data, torch.randn, n_iter=args.niter, p=args.p, e=0.1)
            elif (model_type == 'JWVI'):
                loss = model.compute_join_wasserstein_vi_loss(data, torch.randn, n_iter=args.niter, p=args.p, e=0.1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= (batch_idx + 1)
        print("Epoch: " + str(epoch) + " Loss: " + str(total_loss))

        if (epoch % 1 == 0):
            model.eval()
            sampling(model_dir + '/sample_epoch_' + str(epoch) + ".png", fixednoise, model.decoder, 64, image_size,
                     num_chanel)
            if ( model_type[0] == 'J'):
                for _, (input, y) in enumerate(test_loader, start=0):
                    input = input.to(device)
                    input = input.view(-1, image_size ** 2)
                    reconstruct(model_dir + '/reconstruction_epoch_' + str(epoch) + ".png",
                                input, model.encoder, model.decoder, image_size, num_chanel, device)
                    break
            model.train()


if __name__ == '__main__':
    main()