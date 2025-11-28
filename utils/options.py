"""
    Parse input arguments
"""

import argparse


class Options:
    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser()

        # Training Parameter
        parser.add_argument("--train", action='store_true', help="train or test", default=False)
        parser.add_argument("--BegEpoch", type=int, help="The Begin Epoch", default=1)
        parser.add_argument("--NumEpoch", type=int, help="The Number of Epoch", default=2)
        parser.add_argument("--lr", type=float, help="Learning Rate", default=1e-3)
        parser.add_argument("--num_workers", type=float, help="The number of loader workers", default=16)
        parser.add_argument("--gamma", type=float, help="Learning Rate Scheduler Gamma", default=0.5)
        parser.add_argument("--split_scale", type=float, help="Validation set proportion", default=0.1)
        parser.add_argument("--CropSize", type=int, help="Training image crop size", default=64)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--use_gpus", action='store_true', help="Usage of GPUs", default=True)
        parser.add_argument("--LogPath", type=str, help="The path of log info", default="LogFiles")
        
        parser.add_argument("--ckp_ld", type=str, help="The path of Luminance Deblurring model weight file", default="pretrained/model_LuminDeblur.pth")
        parser.add_argument("--ckp_cc", type=str, help="The path of Color Compensation model weight file", default="pretrained/model_ImgColorNet.pth")

        parser.add_argument("--TrainImgPath", type=str, help="The path of train blurred image", default="../REDS/train/blur")
        parser.add_argument("--TrainEvePath", type=str, help="The path of train event data", default="../REDS/train/evstack")
        parser.add_argument("--TrainGTPath", type=str, help="The path of train sharp image",default="../REDS/train/sharp")
        
        parser.add_argument("--TestImgPath", type=str, help="The path of test blurred image",default="../REDS/val/blur")
        parser.add_argument("--TestEvePath", type=str, help="The path of test event data",default="../REDS/val/evstack")
        parser.add_argument("--TestGTPath", type=str, help="The path of test sharp data",default="../REDS/val/sharp")
        parser.add_argument("--TestSavePath", type=str, help="The saving path of test result",default="results_REDS")
        
        parser.add_argument("--RealImgPath", type=str, help="The path of test real blurred image",
                            default="../EvRGB_Deblur/blur")
        parser.add_argument("--RealEvePath", type=str, help="The path of test real event data",
                            default="../EvRGB_Deblur/event")
        parser.add_argument("--RealGTPath", type=str, help="The path of test real sharp data",
                            default="../EvRGB_Deblur/sharp")
        parser.add_argument("--RealSavePath", type=str, help="The saving path of test real result",
                            default="result_EvRGBDeblur")
        
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
