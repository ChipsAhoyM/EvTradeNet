import torch
import torch.nn as nn
from .BaseBlock import *

class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.net1_conv0 = nn.Conv2d(3, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        self.net1_convs = nn.Sequential(Conv2D(channel, channel, kernel_size=kernel_size),
                                        Conv2D(channel, channel, kernel_size=kernel_size),
                                        Conv2D(channel, channel, kernel_size=kernel_size),
                                        Conv2D(channel, channel, kernel_size=kernel_size))
        self.net1_recon = nn.Conv2d(channel, 4,  kernel_size,
                                    padding=1, padding_mode='replicate')

        
    def forward(self, img):
        conv0 = self.net1_conv0(img)
        conv1 = self.net1_convs(conv0)
        res = self.net1_recon(conv1)
        
        return res[:,:3,:,:], res[:,3:,:,:]