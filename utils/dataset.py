"""
Code for reading the dataset
"""

import os
import os.path
import numpy as np
import random
import torch
import cv2
import glob
import torch.utils.data as udata


"""
Dataset for training
    ImgPath: path of input images
    EvePath: path of events stack (N*H*W)
    GTPath: path of ground truth images
    CropSize: crop size
    scale: scale factor
"""
def sort_ev(evlist):
    t, x, y, p = evlist[:,0], evlist[:,2], evlist[:,1], evlist[:,3]
    index1 = np.argsort(t)
    t = t[index1]
    x = x[index1]
    y = y[index1]
    p = p[index1]
    return t, x, y, p

def read_ev(ev_path, hh, ww):
    sep_c = 8
    evlist = np.loadtxt(ev_path,dtype=np.int32)
    t, x, y, p = sort_ev(evlist)
    sep = (len(t))/ sep_c
    idx = np.array([i//sep for i in range(len(t))])
    idx[idx>(sep_c-1)] = sep_c -1
    p[p==0] = -1
    p = -p
    evframe = np.zeros((sep_c, hh, ww), np.float32)
    np.add.at(evframe, (np.int8(idx),x,y), p)
    return evframe

class Dataset_Train_RGB(udata.Dataset):
    def __init__(self, ImgPath, EvePath, GTPath, CropSize, mode = 'train', reverse = False):
        super(Dataset_Train_RGB, self).__init__()
        self.imgList = sorted(glob.glob(os.path.join(ImgPath, '*.png')))
        self.eveList = sorted(glob.glob(os.path.join(EvePath, '*.npz')))
        self.gtList = sorted(glob.glob(os.path.join(GTPath, '*.png')))
        
        if len(self.imgList) != len(self.eveList) or len(self.imgList) != len(self.gtList):
            raise ValueError("Data is unpaired!")

        self.CropSize = CropSize
        self.reverse = reverse

    def __len__(self):
        return len(self.imgList)

    def MyRandomCrop(self, input, gt, events):
        """
        Random crop for training
        """
        (_, h, w) = events.shape
        i = random.randint(0, h - self.CropSize)
        j = random.randint(0, w - self.CropSize)

        input_patch = input[:, i:i+self.CropSize, j:j+self.CropSize]
        gt_patch = gt[:, i:i + self.CropSize, j :j + self.CropSize]
        event_patch = events[:, i:i + self.CropSize, j:j + self.CropSize]
        return input_patch, gt_patch, event_patch

    def __getitem__(self, index):
        imgs = self.imgList[index]
        gts = self.gtList[index]
        ev = self.eveList[index]
        
        
        imgs = cv2.imread(imgs)
        gts = cv2.imread(gts)
        
        imgs = imgs.transpose(2,0,1)
        gts = gts.transpose(2,0,1)
            
        ev = np.load(ev, allow_pickle=True)['arr_0']
        imgs, gts, ev = self.MyRandomCrop(imgs, gts, ev)
        
        input = torch.Tensor(imgs)/255.0
        gt = torch.Tensor(gts)/255.0
        ev = torch.Tensor(ev)
        
        if self.reverse:
            ev = -ev
        return {'input':input, 'ev':ev, 'gt':gt}

"""
Dataset for testing
    ImgPath: path of input images
    EvePath: path of events stack (N*H*W)
"""
class Dataset_Test_RGB(udata.Dataset):
    def __init__(self, ImgPath, EvePath, GTPath, reverse = False):
        super(Dataset_Test_RGB, self).__init__()
        self.imgList = sorted(glob.glob(os.path.join(ImgPath, '*.png')))
        self.eveList = sorted(glob.glob(os.path.join(EvePath, '*.npz')))
        self.gtList = sorted(glob.glob(os.path.join(GTPath, '*.png')))
        
        print(len(self.imgList), len(self.eveList), len(self.gtList))
        if len(self.imgList) != len(self.eveList) or len(self.imgList) != len(self.gtList):
            raise ValueError("Data is unpaired!")
        
        self.reverse = reverse

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, index):
        imgs = self.imgList[index]
        _, name = os.path.split(imgs)
        gts = self.gtList[index]
        ev = self.eveList[index]
        
        imgs = cv2.imread(imgs)
        gts = cv2.imread(gts)
        
        imgs = cv2.resize(imgs, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)
        gts  = cv2.resize(gts, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)
        
        imgs = imgs.transpose(2,0,1)
        gts = gts.transpose(2,0,1)
            
        ev = np.load(ev, allow_pickle=True)['arr_0']
        
        input = torch.Tensor(imgs)/255.0
        gt = torch.Tensor(gts)/255.0
        ev = torch.Tensor(ev)
            
        return {'input':input, 'ev':ev, 'gt':gt, 'name':name[:-4]}
    
class Dataset_Test_ERD(udata.Dataset):
    def __init__(self, ImgPath, EvePath, GTPath, hh=624, ww=840):
        super(Dataset_Test_ERD, self).__init__()
        self.imgList = sorted(glob.glob(os.path.join(ImgPath, '*.png')))
        self.eveList = sorted(glob.glob(os.path.join(EvePath, '*.txt')))
        self.gtList = sorted(glob.glob(os.path.join(GTPath, '*.png')))
        print(len(self.imgList), len(self.eveList))
        if len(self.imgList) != len(self.eveList):
            raise ValueError("Data is unpaired!")
        
        self.hh = hh
        self.ww = ww

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, index):
        imgs = self.imgList[index]
        ev = self.eveList[index]
        _, name = os.path.split(imgs)

        input = cv2.imread(imgs).transpose(2,0,1)
        gt = cv2.imread(self.gtList[index])
        events = read_ev(ev, self.hh, self.ww)
        
        return {'input':torch.Tensor(input)/255.0, 'ev':torch.Tensor(events), 'gt':gt, 'name':name[:-4]}