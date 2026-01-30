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

# CKPT_PATH = "logs/unet_i2i/hdlmu4ns/checkpoints/unet_efb0_730k.ckpt"
CKPT_PATH = "ckpts/ap2800_dtd5c_50s/unet_efb0_730k.ckpt"

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

def make_masked(img, mask):
    return img * (1 - mask[:, :, None] / 255.0) + mask[:, :, None] / 255.0 * 255.0

def to_tensor(img):
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

def load_images(img_file, mask_file):
    print(img_file, mask_file, IMAGE_DIR, MASKED_DIR, MASK_DIR)
    
    img_filename = img_file#Path(img_file.name).name
    mask_filename = mask_file#Path(mask_file.name).name

    gt_path = Path(img_filename)
    # masked_path = MASKED_DIR / img_filename
    mask_path = Path(mask_filename)
    

    if not gt_path.exists():
        raise gr.Error(f"File not found: {gt_path}")
    if not mask_path.exists():
        raise gr.Error(f"File not found: {mask_path}")

    gt = load_rgb(gt_path)
    # masked = load_rgb(masked_path)
    mask = load_mask(mask_path)
    mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    masked = make_masked(gt, mask)

    diff = diff_heatmap(masked, gt)

    return masked/255, gt/255, mask/255, diff/255


def fix_image(img_file, mask_file):
    start = time.time()

    img_filename = img_file#Path(img_file.name).name
    mask_filename = mask_file#Path(mask_file.name).name

    gt_path = Path(img_filename)
    # masked_path = MASKED_DIR / img_filename
    mask_path = Path(mask_filename)

    # masked = load_rgb(masked_path)
    gt = load_rgb(gt_path)
    mask = load_mask(mask_path)[:, :, None]
    mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    masked = make_masked(gt, mask)

    x = to_tensor(masked).to(DEVICE)

    with torch.no_grad():
        torch.cuda.empty_cache()
        pred = MODEL(x)

    # print(pred.shape)
    mask = np.repeat(np.expand_dims(mask, axis = 2),3, axis=2)
    pred = pred.squeeze(0).repeat(3,1,1).permute(1, 2, 0).cpu().numpy()
    if pred.shape != mask.shape:
        pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        print("Resized pred to match mask shape.")

    psnr = PeakSignalNoiseRatio(data_range=1.0)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    mse_val = np.mean((pred - (mask.astype(np.float32) / 255.0)) ** 2)
    l1_val = np.mean(np.abs(pred - (mask.astype(np.float32) / 255.0)))
    psnr_val = psnr(torch.from_numpy(pred).permute(2,0,1).unsqueeze(0), torch.from_numpy(mask.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)).item()
    ssim_val = ssim(torch.from_numpy(pred).permute(2,0,1).unsqueeze(0), torch.from_numpy(mask.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)).item()
    pred = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    
    # print(pred.shape, mask.shape)

    diff_pred_mask = diff_heatmap(pred, mask)
    diff_pred_img = diff_heatmap(pred, gt)
    # diff_pred = cv2.absdiff(diff_pred_mask, diff_pred_img)
    diff_pred = diff_heatmap(diff_pred_mask, diff_pred_img)
    verdict = ""
    if np.abs(diff_pred_mask).sum() < np.abs(diff_pred_img).sum():
        verdict = "Good. Prediction is close to Mask/Target."
    else:
        verdict = "Bad. Model has learned to copy the input."
    # print(diff_pred.shape, pred.shape)
    # overlay, perc = sift_match_visualize(pred.squeeze(), mask)

    elapsed = time.time() - start
    metrics_text = f"{verdict}\nMSE: {mse_val:.4f}\nL1: {l1_val:.4f}\nPSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.4f}"

    return pred.squeeze(), diff_pred, f"{elapsed:.3f} seconds", metrics_text
