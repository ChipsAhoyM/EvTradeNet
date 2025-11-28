from .BaseBlock import *
import torch
import torch.nn as nn
from .Recon import reconstruct_layer
from .Decompose import DecomNet

class IntergelNet(nn.Module):
    def __init__(self,inc=8):
        super(IntergelNet, self).__init__()
        self.conv_1 = nn.Sequential(
            Conv2D(1, 64, 3),
            Dense_block(64, 16)
        )
        self.conv_2 = nn.Sequential(
            Conv2D(128, 128, 2, 2, padding=0),
            Dense_block(128, 32)
        )
        self.conv_3 = nn.Sequential(
            Conv2D(256, 256, 2, 2, padding=0),
            Dense_block(256, 64)
        )
        
        self.conv_1e = nn.Sequential(
            Conv2D(inc, 64, 3),
            Dense_block(64, 16)
        )
        self.conv_2e = nn.Sequential(
            Conv2D(128, 128, 2, 2, padding=0),
            Dense_block(128, 32)
        )
        self.conv_3e = nn.Sequential(
            Conv2D(256, 256, 2, 2, padding=0),
            Dense_block(256, 64)
        )

        self.fusion = nn.Sequential(
            Conv2D(512*2, 512, 1, padding=0),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )
    # deconv 
        self.deconv_2 = DeConv2D(512, 256,pad=(0,0))

        self.conv_5 = nn.Sequential(
            Conv2D(256*3, 128, 1, padding=0),
            Dense_block(128, 32)
        )

        self.deconv_1 = DeConv2D(256, 128,pad=(0,0))

        self.conv_6 = nn.Sequential(
            Conv2D(128*3, 32, 1, padding=0),
            Dense_block(32, 8)
        )

    # prediction
        self.predConv = nn.Sequential(
            ResidualBlock(channel_num=64),
            ResidualBlock(channel_num=64),
            nn.Conv2d(64, inc, 3, padding=1),
        )
        
    def forward(self, image, event):
        c1 = self.conv_1(image)
        c2 = self.conv_2(c1)
        c3 = self.conv_3(c2)

        lstm = self.conv_1e(event)
        ce1 = self.conv_2e(lstm)
        ce2 = self.conv_3e(ce1)

        m3 = torch.cat([c3, ce2], dim=1)
        fusion = self.fusion(m3)
        
        dc2 = self.deconv_2(fusion)  
        m2 = torch.cat([c2, ce1, dc2], dim=1)
        c5 = self.conv_5(m2)
        
        dc1 = self.deconv_1(c5)
        m1 = torch.cat([c1, lstm, dc1], dim=1)
        c6 = self.conv_6(m1)

        pred = self.predConv(c6)

        return pred

class MergeNet(nn.Module):
    def __init__(self,inc=8,head=4,ffn_expansion_factor=4,bias=True,LayerNorm_type='WithBias',num_blocks=[4],out_channels=1):
        super(MergeNet, self).__init__()
        
        self.block1 = TransformerBlock(dim=inc, num_heads=head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.block2 = CrossTransformerBlock(dim=inc, dim2=1, num_heads=head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.block3 = TransformerBlock(dim=inc, num_heads=head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.block4 = CrossTransformerBlock(dim=inc, dim2=1, num_heads=head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.output1  = nn.Conv2d(inc, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output2 = nn.Conv2d(inc, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.pos_embedding1 = nn.Parameter(torch.randn(1,inc,1,1))
        self.pos_embedding2 = nn.Parameter(torch.randn(1,inc,1,1))
    
    def forward(self, events, img):
        # forwar pass
        feat1 = self.block1(events + self.pos_embedding1)
        feat2 = self.block2(feat1,img)
        feat3 = self.block3(feat2)
        feat4 = self.block4(feat3,img)
        
        log_residual = self.output1(feat4)
        
        # backward pass
        feat5 = self.block1(-events + self.pos_embedding2)
        feat6 = self.block2(feat5,img)
        feat7 = self.block3(feat6)
        feat8 = self.block4(feat7,img)

        back_log_residual = self.output2(feat8)
        
        return log_residual, back_log_residual

    
class EvLuminNet(nn.Module):
    def __init__(self,inc=8):
        super(EvLuminNet, self).__init__()
        self.ENet = IntergelNet(inc=inc)
        self.MNet = MergeNet(inc=inc,head=4,ffn_expansion_factor=4,bias=True,LayerNorm_type='WithBias',num_blocks=[4],out_channels=1)
        self.se = SEBlock(inc,r=2)
        self.recon = reconstruct_layer()
        
        
    def forward(self, image, event):
        pred_log = self.ENet(image, event)
        se_log = self.se(pred_log)
        residual, residual2 = self.MNet(se_log, image)
        restored = image+residual
        restored2 = image+residual2
        
        final = self.recon(restored,restored2)
        
        return final

class LuminDeblur(nn.Module):
    def __init__(self):
        super(LuminDeblur, self).__init__()
        self.decompose = DecomNet()
        self.deblur = EvLuminNet()

    def forward(self, img, ev, gt=None):
        L_img, H_img  = self.decompose(img)
        restored_H = self.deblur(H_img,ev)
        
        if gt != None:
            L_gt, H_gt =self.decompose(gt)
            return {'H_img':H_img, 'L_img':L_img, 'restoredH':restored_H, 'H_gt':H_gt, 'L_gt':L_gt}
        else:
            return {'H_img':H_img, 'L_img':L_img, 'restoredH':restored_H}
 
@torch.jit.script
def torch_laplacian(img_tensor):
    padded = F.pad(img_tensor, pad=[1, 1, 1, 1], mode='reflect')
    return padded[:, :, 2:, 1:-1] + padded[:, :, 0:-2, 1:-1] + padded[:, :, 1:-1, 2:] + padded[:, :, 1:-1, 0:-2] - \
           4 * img_tensor
            
class ImgColorNet(nn.Module):
    def __init__(self):
        super(ImgColorNet, self).__init__()
        self.conv_1 = nn.Sequential(
            Conv2D(3, 64, 3),
            Dense_block(64, 16)
        )
        self.conv_2 = nn.Sequential(
            Conv2D(128, 128, 2, 2, padding=0),
            Dense_block(128, 32)
        )
        self.conv_3 = nn.Sequential(
            Conv2D(256, 256, 2, 2, padding=0),
            Dense_block(256, 64)
        )
        
        self.conv_1e = nn.Sequential(
            Conv2D(8, 64, 3),
            Dense_block(64, 16)
        )
        
        self.conv_1g = nn.Sequential(
            Conv2D(1, 64, 3),
            Dense_block(64, 16)
        )
        
        self.mergeConv = Conv2D(128*2,128,1,padding=0)
        self.seblock = SEBlock(128,r=16)
        
        self.conv_2e = nn.Sequential(
            Conv2D(128, 128, 2, 2, padding=0),
            Dense_block(128, 32)
        )
        self.conv_3e = nn.Sequential(
            Conv2D(256, 256, 2, 2, padding=0),
            Dense_block(256, 64)
        )

        self.fusion = nn.Sequential(
            Conv2D(512*2, 512, 1, padding=0),
            ResidualBlock(512),
        )
    # deconv 
        self.deconv_2 = DeConv2D(512, 256,pad=(0,0))

        self.conv_5 = nn.Sequential(
            Conv2D(256*3, 128, 1, padding=0),
            Dense_block(128, 32)
        )

        self.deconv_1 = DeConv2D(256, 128,pad=(0,0))

        self.conv_6 = nn.Sequential(
            Conv2D(128*3, 32, 1, padding=0),
            Dense_block(32, 8)
        )

    # prediction
        self.predConv = nn.Sequential(
            ResidualBlock(channel_num=64),
            nn.Conv2d(64,3, 3, padding=1),
        )
        
    def forward(self, color, event, dimg):
        c1 = self.conv_1(color)
        c2 = self.conv_2(c1)
        c3 = self.conv_3(c2)

        lstm = self.conv_1e(event)
        gfeat = self.conv_1g(torch_laplacian(dimg))
        init_f = self.seblock(self.mergeConv(torch.cat([lstm, gfeat], dim=1)))
        
        ce1 = self.conv_2e(init_f)
        ce2 = self.conv_3e(ce1)

        m3 = torch.cat([c3, ce2], dim=1)
        fusion = self.fusion(m3)
        
        dc2 = self.deconv_2(fusion)  
        m2 = torch.cat([c2, ce1, dc2], dim=1)
        c5 = self.conv_5(m2)
        
        dc1 = self.deconv_1(c5)
        m1 = torch.cat([c1, init_f, dc1], dim=1)
        c6 = self.conv_6(m1)

        pred = self.predConv(c6)
        out = pred + color
        return out 
        
        
        
        