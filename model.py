import torch.nn as nn 
from functools import reduce
import torch
from torch import nn
from torch import autograd
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# def one(device):
#     return torch.FloatTensor([1]).to(device)

class Generator(nn.Module):
    '''
    Generator model 

    output images with given vectors.
    '''
    def __init__(self, feature_dim=32, out_shape=(1, 32, 32)):
        super(Generator, self).__init__()

        self.feature_dim = feature_dim
        self.feature_size = (out_shape[1] // 8, out_shape[2] // 8)
        self.out_shape = out_shape
        self.resize_layer = nn.Sequential(
            # nn.Linear(feature_dim, 8 * feature_dim * self.feature_size[0] * self.feature_size[1]),
            nn.Linear(feature_dim, 4 * feature_dim * self.feature_size[0] * self.feature_size[1]),
            nn.ReLU()
        )
        self.deconv_layers = nn.ModuleList([
            # nn.Sequential(
            #     nn.ConvTranspose2d(8 * feature_dim, 4 * feature_dim, 4, 2, 1), # kernel_size=4, stride=2, padding=1 => conv: size /= 2, deconv: size *= 2
            #     nn.ReLU(),
            #     nn.BatchNorm2d(4 * feature_dim)
            # ),
            nn.Sequential(
                nn.ConvTranspose2d(4 * feature_dim, 2 * feature_dim, 4, 2, 1), # kernel_size=4, stride=2, padding=1 => conv: size /= 2, deconv: size *= 2
                nn.ReLU(),
                nn.BatchNorm2d(2 * feature_dim)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(2 * feature_dim, 1 * feature_dim, 4, 2, 1), # kernel_size=4, stride=2, padding=1 => conv: size /= 2, deconv: size *= 2
                nn.ReLU(),
                nn.BatchNorm2d(1 * feature_dim)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(feature_dim, self.out_shape[0], 4, 2, 1), # kernel_size=4, stride=2, padding=1 => conv: size /= 2, deconv: size *= 2
                nn.Tanh()
            ),
        ]
        )

    def forward(self, x):
        x = self.resize_layer(x).view(-1, 4 * self.feature_dim, self.feature_size[0], self.feature_size[1])
        # x = self.resize_layer(x).view(-1, 2 * self.feature_dim, self.feature_size[0], self.feature_size[1])
        return reduce(lambda x, l: l(x), self.deconv_layers, x).view(-1, *self.out_shape)
    
    def _train(self, optimizer, criterion, discriminator, fake_data, device):
        optimizer.zero_grad()
        loss = -discriminator(fake_data).mean()
        loss.backward()
        optimizer.step()

        return loss.cpu().detach()

class Discriminator(nn.Module):
    '''
    Discriminator model 

    output images with given vectors.
    '''
    def __init__(self, feature_dim, in_shape=(1, 32, 32), leaky_slope=0.2, drop_out=0.3, lambda_term=10):
        super(Discriminator, self).__init__()
        channels = in_shape[0]

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1 * in_shape[0], 1 * feature_dim, 4, 2, 1),
                nn.LeakyReLU(leaky_slope)
            ),
            nn.Sequential(
                nn.Conv2d(1 * feature_dim, 2 * feature_dim, 4, 2, 1),
                nn.LeakyReLU(leaky_slope)
            ),
            nn.Sequential(
                nn.Conv2d(2 * feature_dim, 4 * feature_dim, 4, 2, 1),
                nn.LeakyReLU(leaky_slope)
            ),
            nn.Sequential(
                nn.Conv2d(4 * feature_dim, 8 * feature_dim, 4, 2, 1),
                nn.LeakyReLU(leaky_slope)
            )
        ])
        self.last_size = in_shape[1] * in_shape[2] * 8 * feature_dim / (16 ** 2)    
        self.resize_layer = nn.Linear(self.last_size, 1)
        self.lambda_term = lambda_term

    def forward(self, x):
        x =  reduce(lambda x, l: l(x), self.conv_layers, x)
        x = x.view(-1, self.last_size)
        return self.resize_layer(x)

    def _train(self, optimizer, criterion, real_data, fake_data, device):
        optimizer.zero_grad()

        loss_real = -self.forward(real_data).mean()
        loss_fake = self.forward(fake_data).mean()

        # gradient penalty
        gradient_penalty = self._calc_gradient_penalty(real_data, fake_data, device) * self.lambda_term

        loss = loss_real + loss_fake + gradient_penalty
        loss.backward()
        optimizer.step() 

        return loss.cpu().detach()

    def _calc_gradient_penalty(self, real_data, fake_data, device):
        '''
        calculate the term of gradient penalty
        '''
        B = real_data.size(0) # batch_size
        C = real_data.size(1) # channels
        H = real_data.size(2) # height
        W = real_data.size(3) # width
 
        eta = torch.FloatTensor(B, 1, 1, 1).uniform_(0, 1).repeat(1, C, H, W).to(device)
        interpolated = (eta * real_data + (1 - eta) * fake_data).to(device)

        interpolated.requires_grad=True
        output = self.forward(interpolated)

        gradients = autograd.grad(output, interpolated, torch.ones(output.size()).to(device), True, True)[0] # calc grad without calling backward won't change the value of param.grad which is still 0
        
        return ((gradients.norm(p=2, dim=1) - 1) ** 2).mean()