#!/usr/bin/env python3
import matplotlib as mpl
mpl.use('Agg') # for SSH service
from argparse import ArgumentParser
import numpy as np
import torch
from model import *
from data import *
from train import *


# parser args
parser = ArgumentParser('Vanilla GAN')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--D-epochs', type=int, default=5)
# parser.add_argument('--G-epochs', type=int, default=1)
# parser.add_argument('--hidden-layers', type=int, default=3)
parser.add_argument('--lr', type=int, default=0.0003)
parser.add_argument('--feature-dim', type=int, default=32)
parser.add_argument('--batch-size', type=int, default=16) # 128
parser.add_argument('--no-gpus', action='store_false', dest='cuda')
parser.add_argument('--test', action='store_true')

if __name__ == '__main__':
    # parse the args
    args = parser.parse_args()

    # whether to use cuda
    cuda = torch.cuda.is_available() and args.cuda

    # dataset
    train_dataset = get_dataset('mnist')
    test_dataset = get_dataset('mnist', train=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # cuda 
    device = torch.device("cuda:0" if torch.cuda.is_available() and cuda else "cpu")
    
    G = Generator(args.feature_dim).to(device)
    D = Discriminator(32).to(device)

    if not args.test:
        train(G, D, args, train_dataloader, device)
    