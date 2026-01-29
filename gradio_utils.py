from pathlib import Path
import numpy as np
import torch
import gradio as gr
import time
from PIL import Image
import cv2
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

import pytorch_lightning as pl
from models.unet_efb0 import UNet_pcb
from config import *

#MODEL RESUME SETUP
class Img2ImgModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model_cfg = ModelConfig_ENB0()
        model = UNet_pcb(pretrained=model_cfg.pretrained,
                         layer1_features=model_cfg.layer1_features,
                         layer2_features=model_cfg.layer2_features,
                         layer3_features=model_cfg.layer3_features,
                         layer4_features=model_cfg.layer4_features,
                         layer5_features=model_cfg.layer5_features)
        self.model = model

    def forward(self, x):
        return self.model(x)

def load_model():
    model = Img2ImgModel.load_from_checkpoint(
        CKPT_PATH,
        map_location=DEVICE,
        strict=False
    )
    model.eval()
    model.to(DEVICE)
    return model

DATASET_ROOT = Path("../art_painting_data/test")
IMAGE_DIR = DATASET_ROOT / "image"
MASKED_DIR = DATASET_ROOT / "masked"
MASK_DIR = DATASET_ROOT / "mask"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = "logs/unet_i2i/puvmf076/checkpoints/unet_efb0_730k.ckpt"

MODEL = load_model()

def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def load_mask(path):
    return np.array(Image.open(path).convert("L"))

def diff_heatmap(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    diff = diff.mean(axis=2)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(diff.astype(np.uint8), cv2.COLORMAP_JET)
    return heatmap

def sift_match_visualize(img1_gray, img2_gray, ratio_thresh=0.75):
    """
    Detect SIFT keypoints and match between two grayscale images.
    Returns:
        overlay_img: RGB image showing matched/unmatched keypoints
        match_percentage: % of keypoints in img1 that have matches in img2
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None:
        # No keypoints detected
        overlay_img = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)
        return overlay_img, 0.0

    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # Create RGB image to draw
    overlay_img = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)

    # Draw matched keypoints in green, unmatched in red
    matched_idx = set([m.queryIdx for m in good_matches])

    for i, kp in enumerate(kp1):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if i in matched_idx:
            color = (0, 255, 0)  # green
        else:
            color = (0, 0, 255)  # red
        cv2.circle(overlay_img, (x, y), 3, color, -1)

    match_percentage = 100 * len(good_matches) / len(kp1) if kp1 else 0.0

    return overlay_img, match_percentage

def to_tensor(img):
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

def load_images(file):
    print(file, IMAGE_DIR, MASKED_DIR, MASK_DIR)
    
    filename = Path(file.name).name

    gt_path = IMAGE_DIR / filename
    masked_path = MASKED_DIR / filename
    mask_path = MASK_DIR / filename

    if not gt_path.exists():
        raise gr.Error(f"File not found: {gt_path}")

    gt = load_rgb(gt_path)
    masked = load_rgb(masked_path)
    mask = load_mask(mask_path)

    diff = diff_heatmap(masked, gt)

    return masked, gt, mask, diff


def fix_image(file):
    start = time.time()

    filename = Path(file.name).name
    masked_path = MASKED_DIR / filename
    gt_path = IMAGE_DIR / filename
    mask_path = MASK_DIR / filename

    masked = load_rgb(masked_path)
    gt = load_rgb(gt_path)
    mask = load_mask(mask_path)[:, :, None]

    x = to_tensor(masked).to(DEVICE)

    with torch.no_grad():
        pred = MODEL(x)

    pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    mse_val = np.mean((pred - (mask.astype(np.float32) / 255.0)) ** 2)
    psnr_val = psnr(torch.from_numpy(pred).permute(2,0,1).unsqueeze(0), torch.from_numpy(mask.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)).item()
    ssim_val = ssim(torch.from_numpy(pred).permute(2,0,1).unsqueeze(0), torch.from_numpy(mask.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)).item()
    pred = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    
    # print(pred.shape, mask.shape)

    diff_pred = diff_heatmap(pred, mask)
    # print(diff_pred.shape, pred.shape)
    overlay, perc = sift_match_visualize(pred.squeeze(), mask)

    elapsed = time.time() - start
    metrics_text = f"MSE: {mse_val:.4f}\nPSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.4f} \nSIFT Match %: {perc:.2f}%"

    return pred.squeeze(), diff_pred, overlay, f"{elapsed:.3f} seconds", metrics_text
