import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

def check_gpu():
    if torch.cuda.is_available():
        print("GPU is available.")
        print("GPU Device Name:", torch.cuda.get_device_name(0))  # Get the name of the first GPU device
        print("GPU Device Count:", torch.cuda.device_count())  # Get the number of available GPUs
        print("CUDA Version:", torch.version.cuda)  # Get the installed CUDA version
        print("PyTorch Version:", torch.__version__)  # Get the installed PyTorch version
    else:
        print("GPU is not available. Using CPU.")

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def show_image_mask_grid(images, masks, nrow=4, img_name = 'imgmask_grid'):
    """
    images: (B, 3, H, W)
    masks:  (B, 1, H, W)
    """
    assert images.shape[0] == masks.shape[0]

    B, _, H, W = images.shape

    # Normalize masks to [0,1] for visualization
    masks_vis = masks.float()
    if masks_vis.max() > 1:
        masks_vis = masks_vis / masks_vis.max().clamp(min=1)

    # Convert masks to 3-channel so grid looks consistent
    masks_vis = masks_vis.repeat(1, 3, 1, 1)

    # Stack image above mask for each sample
    stacked = torch.cat([images, masks_vis], dim=2)
    # stacked shape: (B, 3, 2H, W)

    grid = vutils.make_grid(
        stacked,
        nrow=nrow,
        padding=2,
        normalize=True,
        # value_range=(-1, 1)  # remove if images already [0,1]
    )

    plt.figure(figsize=(12, 6))
    plt.imsave(f'logs/{img_name}.png',grid.permute(1, 2, 0),)
    # plt.axis("off")
    # plt.show()



