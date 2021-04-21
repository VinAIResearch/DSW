from __future__ import print_function

import argparse
import os
import random

import torch
import torchvision.datasets as datasets
from DCGANAE import DCGANAE
from experiments import sampling, sampling_eps
from gsw import GSW
from gswnn import GSW_NN
from PIL import Image
from skimage import io
from torch import optim
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from TransformNet import TransformNet
from utils import circular_function, load_dmodel, save_dmodel

torch.autograd.set_detect_anomaly(True)
# torch.backends.cudnn.enabled = False
class CustomDataset(Dataset):
    def __init__(self, root, image_loader=io.imread, transform=None):
        self.root = root
        self.images_files = os.listdir(root)
        self.loader = image_loader
        self.transform = transform

    def __len__(self):
        # Here, we need to return the number of samples in this dataset.
        return len(self.images_files)

    def __getitem__(self, index):
        img = Image.fromarray(self.loader(self.root + "/" + self.images_files[index]))

        if self.transform is not None:
            images = self.transform(img)
        return (images, 1)


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train", image_loader=io.imread):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.loader = image_loader
        self.files_A = sorted(os.listdir(os.path.join(root, "%sA" % mode)))
        self.files_B = sorted(os.listdir(os.path.join(root, "%sB" % mode)))
        self.rootA = os.path.join(root, "%sA" % mode)
        self.rootB = os.path.join(root, "%sB" % mode)

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.rootA + "/" + self.files_A[index % len(self.files_A)]).convert("RGB"))
        if self.unaligned:
            item_B = self.transform(
                Image.open(self.rootB + "/" + self.files_B[random.randint(0, len(self.files_B) - 1)]).convert("RGB")
            )
        else:
            item_B = self.transform(
                Image.open(self.rootB + "/" + self.files_B[index % len(self.files_B)]).convert("RGB")
            )

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


def main():
    # train args
    parser = argparse.ArgumentParser(description="Disributional Sliced Wasserstein Autoencoder")
    parser.add_argument("--datadir", default="./", help="path to dataset")
    parser.add_argument("--outdir", default="./result", help="directory to output images")
    parser.add_argument(
        "--batch-size", type=int, default=512, metavar="N", help="input batch size for training (default: 512)"
    )
    parser.add_argument(
        "--epochs", type=int, default=200, metavar="N", help="number of epochs to train (default: 200)"
    )
    parser.add_argument("--lr", type=float, default=0.0005, metavar="LR", help="learning rate (default: 0.0005)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        metavar="N",
        help="number of dataloader workers if device is CPU (default: 16)",
    )
    parser.add_argument("--seed", type=int, default=16, metavar="S", help="random seed (default: 16)")
    parser.add_argument("--g", type=str, default="circular", help="g")
    parser.add_argument("--num-projection", type=int, default=1000, help="number projection")
    parser.add_argument("--lam", type=float, default=1, help="Regularization strength")
    parser.add_argument("--p", type=int, default=2, help="Norm p")
    parser.add_argument("--niter", type=int, default=10, help="number of iterations")
    parser.add_argument("--r", type=float, default=1000, help="R")
    parser.add_argument("--latent-size", type=int, default=32, help="Latent size")
    parser.add_argument("--hsize", type=int, default=100, help="Latent size")
    parser.add_argument("--dataset", type=str, default="MNIST", help="(CELEBA|CIFAR)")
    parser.add_argument("--model-type", type=str, required=True, help="(SWD|MSWD|DSWD|GSWD|DGSWD|CRAMER|)")
    parser.add_argument("--cont", type=bool, help="")
    parser.add_argument("--dim", type=int, default=100, help="subspace size")
    parser.add_argument("--e", type=float, default=1000, help="R")
    args = parser.parse_args()
    torch.random.manual_seed(args.seed)
    if args.g == "circular":
        g_function = circular_function
    model_type = args.model_type
    latent_size = args.latent_size
    num_projection = args.num_projection
    dataset = args.dataset
    model_dir = os.path.join(args.outdir, model_type)
    assert dataset in ["CELEBA", "CIFAR", "LSUN"]
    assert model_type in ["SWD", "MSWD", "DSWD", "GSWD", "DGSWD", "MGSWNN"]
    if model_type == "SWD":
        model_dir = os.path.join(args.outdir, model_type + "_n" + str(num_projection))
    elif model_type == "DSWD":
        model_dir = os.path.join(
            args.outdir, model_type + "_iter" + str(args.niter) + "_n" + str(num_projection) + "_lam" + str(args.lam)
        )
    elif model_type == "MSWD":
        model_dir = os.path.join(args.outdir, model_type)
    elif model_type == "MGSWNN":
        model_dir = os.path.join(args.outdir, model_type + "_size" + str(args.hsize))
    elif model_type == "GSWD":
        model_dir = os.path.join(args.outdir, model_type + "_n" + str(num_projection) + "_" + args.g + str(args.r))
    elif model_type == "DGSWD":
        model_dir = os.path.join(
            args.outdir,
            model_type
            + "_iter"
            + str(args.niter)
            + "_n"
            + str(num_projection)
            + "_lam"
            + str(args.lam)
            + "_"
            + args.g
            + str(args.r),
        )
    print(model_dir)
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
    print(
        "batch size {}\nepochs {}\nAdam lr {} \n using device {}\n".format(
            args.batch_size, args.epochs, args.lr, device.type
        )
    )

    if dataset == "CIFAR":
        from DCGANAE import Discriminator

        image_size = 64
        num_chanel = 3
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                args.datadir,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Resize(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    elif dataset == "LSUN":
        from DCGANAE import Discriminator

        image_size = 64
        num_chanel = 3
        train_loader = torch.utils.data.DataLoader(
            datasets.LSUN(
                args.datadir + "/lsun",
                classes=["bedroom_train"],
                transform=transforms.Compose(
                    [
                        transforms.Resize(64),
                        transforms.CenterCrop(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    elif dataset == "CELEBA":
        from DCGANAE import Discriminator

        image_size = 64
        num_chanel = 3
        dataset = CustomDataset(
            root=args.datadir + "/img_align_celeba",
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        # Create the dataloader
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
        # test_loader = torch.utils.data.DataLoader(
        #     dataset, batch_size=1000, shuffle=True, num_workers=args.num_workers, pin_memory=True
        # )

    model = DCGANAE(image_size=64, latent_size=latent_size, num_chanel=3, hidden_chanels=64, device=device).to(device)
    dis = Discriminator(64, args.latent_size, 3, 64).to(device)
    disoptimizer = optim.Adam(dis.parameters(), lr=args.lr, betas=(0.5, 0.999))
    if model_type == "DSWD" or model_type == "DGSWD":
        transform_net = TransformNet(64 * 8 * 4 * 4).to(device)
        op_trannet = optim.Adam(transform_net.parameters(), lr=args.lr, betas=(0.5, 0.999))
        # train_net(64 * 8 * 4 * 4, 1000, transform_net, op_trannet)
    if model_type == "MGSWNN":
        gsw = GSW_NN(din=64 * 8 * 4 * 4, nofprojections=1, model_depth=3, num_filters=args.hsize, use_cuda=True)
    if model_type == "MSWD":
        gsw = GSW()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    epoch_cont = 0
    if args.cont:
        epoch_cont, modelstate, optimizerstate, tnetstate, optnetstate, disstate, opdistate = load_dmodel(model_dir)
        model.load_state_dict(modelstate)
        optimizer.load_state_dict(optimizerstate)
        dis.load_state_dict(disstate)
        disoptimizer.load_state_dict(opdistate)
        epoch_cont = epoch_cont + 1
        print("Continue from epoch " + str(epoch_cont))
    fixednoise = torch.randn((64, latent_size)).to(device)

    for epoch in range(epoch_cont, args.epochs):
        total_loss = 0.0

        for batch_idx, (data, y) in tqdm(enumerate(train_loader, start=0)):
            if model_type == "SWD":
                loss = model.compute_loss_SWD(dis, disoptimizer, data, torch.randn, num_projection, p=args.p)
            elif model_type == "GSWD":
                loss = model.compute_loss_GSWD(
                    dis, disoptimizer, data, torch.randn, g_function, args.r, num_projection, p=args.p
                )
            elif model_type == "MGSWNN":
                loss = model.compute_loss_MGSWNN(dis, disoptimizer, data, torch.randn, gsw, p=args.p)
            elif model_type == "MSWD":
                loss = model.compute_loss_MSWD(dis, disoptimizer, data, torch.randn, gsw)
            elif model_type == "DSWD":
                loss = model.compute_lossDSWD(
                    dis,
                    disoptimizer,
                    data,
                    torch.randn,
                    num_projection,
                    transform_net,
                    op_trannet,
                    p=args.p,
                    max_iter=args.niter,
                    lam=args.lam,
                )
            elif model_type == "DGSWD":
                loss = model.compute_lossDGSWD(
                    dis,
                    disoptimizer,
                    data,
                    torch.randn,
                    num_projection,
                    transform_net,
                    op_trannet,
                    g_function,
                    args.r,
                    p=args.p,
                    max_iter=args.niter,
                    lam=args.lam,
                )
            optimizer.zero_grad()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        total_loss /= batch_idx + 1
        print("Epoch: " + str(epoch) + " Loss: " + str(total_loss))
        save_dmodel(model, optimizer, dis, disoptimizer, None, None, epoch, model_dir)
        sampling(
            model_dir + "/sample_epoch_" + str(epoch) + ".png", fixednoise, model.decoder, 64, image_size, num_chanel
        )
        if epoch == args.epochs - 1:
            model.eval()
            sampling_eps(
                model_dir + "/sample_epoch_" + str(epoch), fixednoise, model.decoder, 64, image_size, num_chanel
            )
            model.train()


if __name__ == "__main__":
    main()
