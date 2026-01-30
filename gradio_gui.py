import gradio as gr
import torch
import time
import numpy as np
import cv2
from PIL import Image

import matplotlib.pyplot as plt

from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from gradio_utils import *
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------

image_dir_state = gr.State(IMAGE_DIR)
masked_dir_state = gr.State(MASKED_DIR)
mask_dir_state = gr.State(MASK_DIR)

model_state = gr.State(lambda: MODEL)
device_state = gr.State(DEVICE)

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Image Restoration Demo (Lightning + UNet)")

    with gr.Row():
        img_file_input = gr.File(label="Drop image file")
        mask_file_input = gr.File(label="Drop mask file")

    with gr.Row():
        gt_img = gr.Image(label="Image")
        mask_img = gr.Image(label="Ground Truth Mask")
        masked_img = gr.Image(label="Masked Image")

    diff_map = gr.Image(label="Difference Heatmap (Masked vs GT)")

    load_btn = gr.Button("Load Images")

    load_btn.click(
        load_images,
        inputs=[img_file_input, mask_file_input],
        outputs=[masked_img, gt_img, mask_img, diff_map]
    )

    gr.Markdown("---")

    fix_btn = gr.Button("Fix Image")

    with gr.Row():
        pred_img = gr.Image(label="Predicted Image")
        pred_diff = gr.Image(label="Difference between difference heat maps of (input and prediction) to (gt and prediction)")
        # overlay = gr.Image(label="SIFT Match Overlay")

    time_text = gr.Textbox(label="Inference Time")
    device_used = gr.Textbox(label="Device Used", value=DEVICE)
    metrics_text = gr.Textbox(label="Metrics (MSE, L1, PSNR, SSIM)")
    
    fix_btn.click(
        fix_image,
        inputs=[img_file_input, mask_file_input],
        outputs=[pred_img, pred_diff, time_text, metrics_text]
    )

demo.launch()