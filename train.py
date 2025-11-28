import os
import time
import torch
import numpy as np
import skimage.metrics
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import LuminDeblur, ImgColorNet
from utils import Options, Dataset_Train_RGB, init_model
import warnings
warnings.filterwarnings('ignore')
from tensorboardX import SummaryWriter
import lpips
import cv2


args = Options().parse()

if args.use_gpus:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

fn_vgg = lpips.LPIPS(net='vgg').to(device)

job_time = time.strftime("%m_%d_%H_%M")
path_log = os.path.join(args.LogPath, job_time)
logger = SummaryWriter(path_log)
os.makedirs(path_log, exist_ok=True)

def norm(x):
    return (x - x.min()) / (x.max() - x.min())


def loss(outputs, gt):
    alpha = 0.5
    beta = 10
    p_loss = torch.nn.MSELoss()(outputs, gt)
    p_f_loss = fn_vgg(outputs*2.0-1.0, gt*2.0-1.0).mean()

    content_loss = alpha * p_f_loss + beta * p_loss
    return content_loss, p_f_loss, p_loss


def make_dataset():
    all_dataset = Dataset_Train_RGB(args.TrainImgPath, args.TrainEvePath, args.TrainGTPath, args.CropSize, mode='train')
    val_dataset = Dataset_Train_RGB(args.TestImgPath, args.TestEvePath, args.TestGTPath, args.CropSize,mode='val')
    train_loader = DataLoader(all_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True, num_workers=args.num_workers)
    return train_loader, val_loader


def train_stage_ld(): 
    model = LuminDeblur() 
    model = model.to(device)
    model = init_model(model)
    
    criterion = loss
    criterion2 = torch.nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10,T_mult=2, eta_min=1e-5)
    model.train()

    train_loader, val_loader = make_dataset()
    bestPNSR = 0.0
    
    for epoch in range(args.BegEpoch, args.NumEpoch + 1):
        epoch_loss = 0
        all_loss1 = 0
        mse_loss = 0
        perc_loss = 0
        all_loss2 = 0
        all_loss3 = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{args.NumEpoch}', dynamic_ncols=True) as pbar:
            for index, data in enumerate(train_loader):
                imgs = data['input'].to(device)
                event = data['ev'].to(device)
                gt = data['gt'].to(device)

                outs = model(imgs, event, gt)
                H_img = outs['H_img']
                L_img = outs['L_img']
                restoredH = outs['restoredH']
                H_gt = outs['H_gt']
                L_gt = outs['L_gt']
                
                loss1, p_f_loss, p_loss = criterion(restoredH*L_img, gt)
                loss2 = criterion2(H_img*L_img,imgs) + criterion2(H_gt*L_gt,gt)
                loss3 = criterion2(H_img*L_gt, imgs) + criterion2(H_gt*L_img,gt)
                loss4 = criterion2(restoredH, H_gt)
                
                Loss = loss1  +  0.1 * (loss2  + loss3)  + loss4 
                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()

                epoch_loss += Loss.item()
                mse_loss += p_loss.item()
                perc_loss += p_f_loss.item()
                
                all_loss1 += loss1.item()
                all_loss2 += loss2.item()
                all_loss3 += loss3.item()
                pbar.set_postfix(**{'loss (batch)': Loss.item(), 'AveLoss': epoch_loss / (index + 1)})
                pbar.update()
        torch.save(model.state_dict(), f'pretrained/model_LuminDeblur_latest.pth')
        scheduler.step()

        AvePSNR, AveSSIM = validation_stage_ld(model, val_loader, epoch, logger)
        if AvePSNR > bestPNSR:
            torch.save(model.state_dict(), f'pretrained/model_LuminDeblur_best.pth')
            bestPNSR = AvePSNR
        logger.add_scalar('Metric-S1/val psnr', AvePSNR, epoch)
        logger.add_scalar('Metric-S1/val ssim', AveSSIM, epoch)

        logger.add_scalar('Loss-S1/loss', epoch_loss/len(train_loader), epoch)
        logger.add_scalar('Loss-S1/mse_loss', mse_loss/len(train_loader), epoch)
        logger.add_scalar('Loss-S1/perc_loss', perc_loss/len(train_loader), epoch)
        logger.add_scalar('Loss-S1/loss1', all_loss1/len(train_loader), epoch)
        logger.add_scalar('Loss-S1/loss2', all_loss2/len(train_loader), epoch)
        logger.add_scalar('Loss-S1/loss3', all_loss3/len(train_loader), epoch)
        logger.add_scalar('LR-S1/lr', optimizer.param_groups[0]['lr'], epoch)

def validation_stage_ld(model, val_loader, epoch, logger):
    model.eval()
    totally_psnr = 0.
    totally_ssim = 0.

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Epoch {epoch} validation', dynamic_ncols=True) as pbar:
            for index, data in enumerate(val_loader):
                imgs = data['input'].to(device)
                event = data['ev'].to(device)
                
                outs = model(imgs, event)
                L_img = outs['L_img']
                restoredH = outs['restoredH']

                restored_img = torch.clamp(restoredH*L_img, 0, 1)
                
                y_blur = np.uint8(torch.squeeze(imgs.cpu()).numpy()*255.0)
                img_z = np.uint8(torch.squeeze(restored_img.cpu()).numpy()*255.0)
                y_gt = np.uint8(torch.squeeze(data['gt']).numpy()*255.0)
                
                psnr = sum([skimage.metrics.peak_signal_noise_ratio(img_z[i,:, :, :], y_gt[i,:, :, :])
                            for i in range(img_z.shape[0])]) / img_z.shape[0]
                ssim = sum([skimage.metrics.structural_similarity(img_z[i,:, :, :].transpose(1,2,0), y_gt[i,:, :, :].transpose(1,2,0), channel_axis=2)
                            for i in range(img_z.shape[0])]) / img_z.shape[0]
                
                if index%10 == 0:
                    # stack N imgs along the x-axis
                    img_restored = np.concatenate([(img_z[i,:,:,:]) for i in range(img_z.shape[0])], axis=2).transpose(1,2,0)
                    img_blur = np.concatenate([(y_blur[i,:,:,:]) for i in range(img_z.shape[0])], axis=2).transpose(1,2,0)
                    img_sharp = np.concatenate([(y_gt[i,:,:,:]) for i in range(img_z.shape[0])], axis=2).transpose(1,2,0)
                    
                    ev_np = np.uint8(norm(np.sum(event.cpu().numpy(),axis=1))*255.0)
                    img_ev = np.concatenate([ev_np[i,:,:] for i in range(ev_np.shape[0])], axis=1)
                    img_ev = np.stack([img_ev]*3, axis=2)
                    
                    vis_L = np.uint8(norm(L_img).cpu().numpy()*255.0)
                    img_L = np.concatenate([vis_L[i,:,:,:] for i in range(vis_L.shape[0])], axis=2).transpose(1,2,0)
                    
                    vis_H = np.uint8(norm(restoredH).cpu().numpy()*255.0)
                    img_H = np.concatenate([vis_H[i,:,:,:] for i in range(vis_H.shape[0])], axis=2)
                    img_H = np.concatenate([img_H]*3, axis=0).transpose(1,2,0)
                    
                    # concat three images along the y-axis
                    img_restored = np.concatenate([img_restored, img_sharp, img_blur,img_H, img_L, img_ev], axis=0)
                    # convert to rgb 
                    img_restored = cv2.cvtColor(img_restored, cv2.COLOR_BGR2RGB)
                    logger.add_image('%02d'%(index//10), img_restored, epoch, dataformats='HWC')


                totally_psnr += psnr
                totally_ssim += ssim
                pbar.set_postfix(**{'psnr': psnr, 'ssim': ssim, 'AvePSNR': totally_psnr / (index + 1),
                                    'AveSSIM': totally_ssim / (index + 1)})
                pbar.update()
    model.train()

    return totally_psnr / len(val_loader), totally_ssim / len(val_loader)

def train_stage_cc(): 
    model = LuminDeblur()
    model1 = ImgColorNet()
    if args.use_gpus:
        model = model.to(device)
        model1 = model1.to(device)
        
    model.load_state_dict(torch.load('pretrained/model_LuminDeblur_best.pth', map_location='cpu'))
    model1 = init_model(model1)
    criterion = loss
    optimizer = optim.AdamW(model1.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100,T_mult=2, eta_min=1e-6)
    model.eval()
    model1.train()

    train_loader, val_loader = make_dataset()
    bestPNSR = 0.0
    
    for epoch in range(args.BegEpoch, args.NumEpoch + 1):
        epoch_loss = 0
        mse_loss = 0
        perc_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{args.NumEpoch}', dynamic_ncols=True) as pbar:
            for index, data in enumerate(train_loader):
                imgs = data['input'].to(device)
                event = data['ev'].to(device)
                gt = data['gt'].to(device)
                
                with torch.no_grad():
                    outs = model(imgs, event, gt)
                L_img = outs['L_img']
                restoredH = outs['restoredH']
                refine_L = model1(L_img, event, restoredH)
                Loss, p_f_loss, p_loss = criterion(restoredH * refine_L, gt)
                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()

                epoch_loss += Loss.item()
                mse_loss += p_loss.item()
                perc_loss += p_f_loss.item()
                

                pbar.set_postfix(**{'loss (batch)': Loss.item(), 'AveLoss': epoch_loss / (index + 1)})
                pbar.update()
        torch.save(model1.state_dict(), f'pretrained/model_ImgColorNet_latest.pth')
        scheduler.step()

        AvePSNR, AveSSIM = validation_stage_cc(model, model1, val_loader, epoch, logger)
        if AvePSNR > bestPNSR:
            torch.save(model1.state_dict(), f'pretrained/model_ImgColorNet_best.pth')
            bestPNSR = AvePSNR
        logger.add_scalar('Metric-S2/val psnr', AvePSNR, epoch)
        logger.add_scalar('Metric-S2/val ssim', AveSSIM, epoch)

        logger.add_scalar('Loss-S2/loss', epoch_loss/len(train_loader), epoch)
        logger.add_scalar('Loss-S2/mse_loss', mse_loss/len(train_loader), epoch)
        logger.add_scalar('Loss-S2/perc_loss', perc_loss/len(train_loader), epoch)
        logger.add_scalar('LR-S2/lr', optimizer.param_groups[0]['lr'], epoch)

def validation_stage_cc(model, model1, val_loader, epoch, logger):
    model1.eval()
    totally_psnr = 0.
    totally_ssim = 0.

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Epoch {epoch} validation', dynamic_ncols=True) as pbar:
            for index, data in enumerate(val_loader):
                imgs = data['input'].to(device)
                event = data['ev'].to(device)
                
                outs = model(imgs, event)
                L_img = outs['L_img']
                restoredH = outs['restoredH']
                refine_L = model1(L_img, event, restoredH)
        
                restored_img = torch.clamp(restoredH * refine_L, 0, 1)
                y_blur = np.uint8(torch.squeeze(imgs.cpu()).numpy()*255.0)
                img_z = np.uint8(torch.squeeze(restored_img.cpu()).numpy()*255.0)
                y_gt = np.uint8(torch.squeeze(data['gt']).numpy()*255.0)
                
                psnr = sum([skimage.metrics.peak_signal_noise_ratio(img_z[i,:, :, :], y_gt[i,:, :, :])
                            for i in range(img_z.shape[0])]) / img_z.shape[0]
                ssim = sum([skimage.metrics.structural_similarity(img_z[i,:, :, :].transpose(1,2,0), y_gt[i,:, :, :].transpose(1,2,0), channel_axis=2)
                            for i in range(img_z.shape[0])]) / img_z.shape[0]
                
                if index%10 == 0:
                    # stack N imgs along the x-axis
                    img_restored = np.concatenate([(img_z[i,:,:,:]) for i in range(img_z.shape[0])], axis=2).transpose(1,2,0)
                    img_blur = np.concatenate([(y_blur[i,:,:,:]) for i in range(img_z.shape[0])], axis=2).transpose(1,2,0)
                    img_sharp = np.concatenate([(y_gt[i,:,:,:]) for i in range(img_z.shape[0])], axis=2).transpose(1,2,0)
                    
                    ev_np = np.uint8(norm(np.sum(event.cpu().numpy(),axis=1))*255.0)
                    img_ev = np.concatenate([ev_np[i,:,:] for i in range(ev_np.shape[0])], axis=1)
                    img_ev = np.stack([img_ev]*3, axis=2)
                    
                    
                    vis_L = np.uint8(norm(torch.exp(L_img)).cpu().numpy()*255.0)
                    img_L = np.concatenate([vis_L[i,:,:,:] for i in range(vis_L.shape[0])], axis=2).transpose(1,2,0)
                    
                    vis_H = np.uint8(norm(restoredH).cpu().numpy()*255.0)
                    img_H = np.concatenate([vis_H[i,:,:,:] for i in range(vis_H.shape[0])], axis=2)
                    img_H = np.concatenate([img_H]*3, axis=0).transpose(1,2,0)

                    img_restored = np.concatenate([img_restored, img_sharp, img_blur,img_H, img_L, img_ev], axis=0)
                    # convert to rgb 
                    img_restored = cv2.cvtColor(img_restored, cv2.COLOR_BGR2RGB)
                    logger.add_image('%02d'%(index//10), img_restored, epoch, dataformats='HWC')


                totally_psnr += psnr
                totally_ssim += ssim
                pbar.set_postfix(**{'psnr': psnr, 'ssim': ssim, 'AvePSNR': totally_psnr / (index + 1),
                                    'AveSSIM': totally_ssim / (index + 1)})
                pbar.update()
    model1.train()
    return totally_psnr / len(val_loader), totally_ssim / len(val_loader)

if __name__ == '__main__':
    train_stage_ld()
    train_stage_cc()
