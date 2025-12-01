# EvTradeNet (ICCVw 2025)

Implementation of **Monochromatic Event Guided Image Deblurring with Event-triggering-aware Decomposition** (ICCV 2025 MIPI Workshop)

## Repository Layout
- `model/`: network building blocks (`BaseBlock.py`) and the three main sub-networks (`Decompose.py`, `EvTradeNet.py`, `Recon.py`).
- `utils/`: dataset readers, option parser, and weight initialization utilities.
- `pretrained/`: placeholder weights for the luminance (`model_LuminDeblur.pth`) and color (`model_ImgColorNet.pth`) stages.
- `train.py` / `test.py`: end-to-end scripts for supervised training and evaluation.

## Environment
```bash
conda create -n evtradenet python=3.10
conda activate evtradenet
pip install torch torchvision
pip install einops numpy opencv-python scikit-image scipy tensorboardX tqdm lpips "huggingface_hub[cli]"
```

## Data Preparation
### Real-world evaluation (EvRGB-Deblur)


The EvRGB-Deblur dataset is hosted on Google Drive.

- Download link: [EvRGB-Deblur Dataset]()
- After downloading, extract the contents under your project directory so the structure matches shown below.
```
EvRGB_Deblur/
  blur/*.png
  event/*.txt
  sharp/*.png    
```

## Train

```shell
python train.py --TrainImgPath `Path of Train Image` --TrainEvePath `Path of Events` --TrainGTPath `Path of Ground Truth Image` --TestImgPath `Path of Test Image` --TestEvePath `Path of Test Events` --TestGTPath `Path of Test Ground Truth Image`
```


## Evaluation
Pretrained weights for both the luminance and color stages are also available on Google Drive.

- Download link: [EvTradeNet Pretrained Weights](https://drive.google.com/drive/folders/19bmHJDD4pHrRRzH0l-8ExD2dWZDFSDzh?usp=sharing)
- Place `model_LuminDeblur_best.pth` and `model_ImgColorNet_best.pth` into the `pretrained/` directory.

```bash
python test.py \
  --ckp_ld pretrained/model_LuminDeblur_best.pth \
  --ckp_cc pretrained/model_ImgColorNet_best.pth \
  --RealImgPath EvRGB_Deblur/blur \
  --RealEvePath EvRGB_Deblur/event \
  --RealGTPath EvRGB_Deblur/sharp \
  --RealSavePath results_evtradenet
```
The script writes restored frames to `RealSavePath`


## References
```bibtex
@InProceedings{Teng_2025_ICCV,
    author    = {Teng, Minggui and Li, Boyu and Yang, Yixin and Zhou, Chu and Chen, Yan and Ren, Jimmy S. and Shi, Boxin},
    title     = {Monochromatic Event Guided Image Deblurring with Event-triggering-aware Decomposition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
    year      = {2025},
    pages     = {3876-3885}
}
```

