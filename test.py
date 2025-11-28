import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import LuminDeblur, ImgColorNet
from utils import Options, Dataset_Test_ERD
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

args = Options().parse()
if args.use_gpus:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def test():
    os.makedirs(args.RealSavePath, exist_ok=True)
    stage_ld = LuminDeblur()
    stage_cc = ImgColorNet()
    if args.use_gpus:
        stage_ld = stage_ld.to(device)
        stage_cc = stage_cc.to(device)

    stage_ld.load_state_dict(torch.load(args.ckp_ld, map_location='cpu'))
    stage_cc.load_state_dict(torch.load(args.ckp_cc, map_location='cpu'))
    
    stage_ld.eval()
    stage_cc.eval()

    test_dataset = Dataset_Test_ERD(args.RealImgPath, args.RealEvePath, args.RealGTPath)
    test_loader = DataLoader(test_dataset)

    metric_psnr = 0
    metric_ssim = 0 
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for data in test_loader:
                imgs = data['input'].to(device)
                event = data['ev'].to(device)
                gt = np.squeeze(data['gt'].numpy())
                outs = stage_ld(imgs, event)
                L_img = outs['L_img']
                restoredH = outs['restoredH']
                refine_L = stage_cc(L_img, event, restoredH)
                restored_img = torch.clamp(restoredH*refine_L, 0, 1)
                img_z = np.uint8(torch.squeeze(restored_img.cpu()).numpy()*255.0).transpose(1,2,0)
                cv2.imwrite(os.path.join(args.RealSavePath, data['name'][0] + f".png"), img_z)
                metric_psnr += peak_signal_noise_ratio(gt, img_z)
                metric_ssim += structural_similarity(gt, img_z, channel_axis=2)
                
                pbar.update()
                pbar.set_postfix_str(f"Average PSNR is {metric_psnr/(pbar.n):.2f}, Average SSIM is {metric_ssim/(pbar.n):.4f}")

if __name__ == "__main__":
    test()