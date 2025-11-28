from .BaseBlock import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange

class reconstruct_layer(nn.Module):
    def __init__(self,inc=1):
        super(reconstruct_layer, self).__init__()
        self.extract = Conv2D(inc, 64, 3)
        self.ps = nn.PixelShuffle(2)
        self.down = Conv2D(32, 64, 2, 2, padding=0)
        self.extract2 = Conv2D(64, 128, 3)
        self.down2 = Conv2D(32, 64, 2, 2, padding=0)
        self.predict = nn.Conv2d(64, 1, 3, padding=1)
    def forward(self,img1,img2):
        e1 = self.extract(img1)
        e2 = self.extract(img2)
        
        fuse = torch.cat([e1,e2],dim=1)
        ps1 = self.ps(fuse)
        dn1 = self.down(ps1)
        ext = self.extract2(dn1)
        ps2 = self.ps(ext)
        dn2 = self.down2(ps2)
        img = self.predict(dn2)
        return img