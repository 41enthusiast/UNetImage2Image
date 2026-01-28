import logging
import numpy as np
import torch
from PIL import Image
# from functools import lru_cache
from functools import partial
# from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config import DataConfig
from albumentations.pytorch import ToTensorV2
import albumentations as A
from utils import show_image_mask_grid
import random

def load_image(filename, mode='RGB'):
    # ext = splitext(filename)[1]
    # if ext == '.npy':
    #     return Image.fromarray(np.load(filename))
    # elif ext in ['.pt', '.pth']:
    #     return Image.fromarray(torch.load(filename).numpy())
    # else:
    return Image.open(filename).convert(mode)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, img_size=None, mask_suffix: str = '', split = 'train'):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.img_size = [img_size, img_size]
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        self.mask_ids = [splitext(file)[0] for file in listdir(mask_dir) if isfile(join(mask_dir, file)) and not file.startswith('.')]
        # self.mask_ids = self.mask_ids*(len(self.ids)//len(self.mask_ids)+1)[:len(self.ids)] #repeat mask ids if less masks than images
        
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        # with Pool() as p:
        #     unique = list(tqdm(
        #         p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
        #         total=len(self.ids)
        #     ))

        # self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        # logging.info(f'Unique mask values: {self.mask_values}')

        def val_tfm(img_size):
            return A.Compose([
                # A.Resize(height=img_size[0], width=img_size[1]),
                # A.Normalize(mean=(0.,), std=(1.0,)),
                ToTensorV2()
            ])

        self.transform = self.data_tfm(self.img_size)
        self.val_transform = val_tfm(self.img_size)
        self.split = split

        # print('New image size:', self.img_size)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def data_tfm(img_size = None):
        """
            implementing the same spatial transformations for both image and mask.
            Then optionally implementing any color intensity changes to only the image
        """
        tfm_prob = 0.5
        resize_range_min = 0.8
        if img_size is [None, None]:
            img_size = [256, 256]
        # print(img_size)
        # img_size = [int(img_size[0]), int(img_size[1])]
        return A.Compose(
            [
                #Flips
                A.HorizontalFlip(p=tfm_prob),
                A.VerticalFlip(p=tfm_prob),
                #Rotate
                A.Rotate(
                    limit=30,
                    interpolation=4,      # cv2.LANCZOS4 (uniformity) for image
                    border_mode=0,        # cv2.BORDER_CONSTANT for now. reflect maybe according to some img gen methods
                    # value=0,
                    # mask_value=0,
                    p=tfm_prob
                ),
                #Resize Crop
                # A.RandomResizedCrop(
                #     size=img_size,
                #     scale=(resize_range_min, 1.0),
                #     ratio=(1.0, 1.0),#dont change the aspect ratio
                #     p=1.0,
                #     interpolation=4
                # ),
                #Normalize - disabling for now
                # A.Normalize(mean=(0.,), std=(1.0,)),
                ToTensorV2()
            ]
        )
        

    @staticmethod
    def preprocess(pil_img, scale, img_size=None):
        w, h = pil_img.size
        if img_size is not None:
            newW, newH = img_size
        else:
            newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        
        pil_img = pil_img.resize((newW, newH), resample=Image.Resampling.LANCZOS)
        img = np.asarray(pil_img)

        # else:
        if img.ndim == 2:
            img = img[..., np.newaxis]
        # else:
        #     img = img.transpose((2, 0, 1))

        if (img > 1).any():
            img = img / 255.0

        return img

    @staticmethod
    def overlay_mask_on_image(img, mask):

        original_image_rgba = img.convert("RGBA")
        newW, newH = original_image_rgba.size
        alpha_channel = np.array(mask.resize((newW, newH), resample=Image.Resampling.LANCZOS))

        # alpha_channel = np.where(alpha_channel > 0, alpha_channel, 255)  # makes this black parts of the mask fully opaque overlays on the background

        # Combine the original image with the new alpha channel
        original_array_rgba = np.array(original_image_rgba)
        original_array_rgba[..., 3] = alpha_channel #image masked with transparent mask here

        def rgba_to_rgb(images_rgba):
            # Assume the background is white (255, 255, 255)
            background = np.ones_like(images_rgba[:, :, :3]) * 255.0  # Shape: [H, W, 3], make opaque mask areas white

            # Alpha factor for blending
            alpha_factor = images_rgba[:,:, 3] / 255.0  # Normalize alpha to [0, 1]

            # Blend the images with the background
            images_rgb = (background * alpha_factor[:, :, None] +
                                images_rgba[:, :, :3] * (1 - alpha_factor[:,  :, None])).astype(np.uint8)

            return images_rgb

        img_rgb = rgba_to_rgb(original_array_rgba)
        img_rgb = Image.fromarray(img_rgb)
        return img_rgb

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob('*.*'))
        # random.shuffle(mask_file)
        mask_file = mask_file[idx % len(mask_file)]
        img_file = list(self.images_dir.glob(name + '.*'))#image folder

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        # assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file, mode='L')
        img = load_image(img_file[0])
        masked = self.overlay_mask_on_image(img, mask)

        img = self.preprocess(masked, self.scale, self.img_size)
        mask = self.preprocess(mask, self.scale, self.img_size)
        
        assert img.size//3 == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size//3} and {mask.size}'

        if self.split == 'train':
            augmented = self.transform(image = img, mask=mask)
        else:
            augmented = {'image': torch.as_tensor(img).permute(2,0,1),
                         'mask': torch.as_tensor(mask)
                        }
            augmented = self.val_transform(image = img, mask=mask)
        return {
            'image': augmented['image'].float().contiguous(),
            'mask':augmented['mask'].permute(2,0,1).float().contiguous()
        }

import ptwt #pywt
import numpy as np
from torch import Tensor

def adaptive_detail_change(images: Tensor, wavelet='db4', levels=3, progressive_thresholding_lvl = 1):
    # Decompose batch of black and white images
    coeffs = ptwt.wavedec2(images, wavelet, level=levels)
    print(coeffs.shape)
    # Keep ONLY approximation (lowest freq), zero all details
    coeffs[1:] = [torch.zeros_like(c) for c in coeffs[progressive_thresholding_lvl:]]
    # Reconstruct
    recon_imgs = ptwt.waverec2(coeffs, wavelet)
    
    return recon_imgs

def adaptive_coverage_change(images: Tensor, split_idx: int, num_splits: int = 3):
    """
    Fast vectorized PyTorch implementation of adaptive coverage change.
    Operates on a batch of images [B, H, W] directly on the current device.
    """
    # 1. Validation (Optional: remove in production for max speed)
    if images.ndim != 3:
        raise ValueError("Expecting a 3D grayscale tensor batch [B, H, W].")
    
    # 2. Compute split bounds
    split_size = 1.0 / num_splits
    lower = split_idx * split_size
    upper = (split_idx + 1) * split_size

    # 3. Sample threshold (Use torch.distributions for device-consistency)
    # We sample a single scalar threshold for the whole batch to match your logic
    thr = (torch.rand(1, device=images.device) * (upper - lower)) + lower
    # 4. Vectorized Thresholding
    out = images.clone()
    # Efficiently set values > thr to 0.0 across the whole batch
    out[out > thr] = 0.0
    
    return out, thr.item()

import tarfile
from torchvision import transforms
from torch.utils.data import IterableDataset
import pytorch_lightning as pl
import io

class TextureAnalysisDataset(IterableDataset):
    def __init__(self, tar_path, wavelet='db4'):
        self.tar_path = tar_path
        self.wavelet = wavelet
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def compute_detail_level(self, img_tensor):
        """Computes DB4 Wavelet Energy as a quantitative stat."""
        # Add batch dim [1, 1, H, W]
        coeffs = ptwt.wavedec2(img_tensor.unsqueeze(0), self.wavelet, level=1)
        deets = coeffs[1] #edges and (details+noise)
        detail_score = torch.stack([(d**2).mean() for d in deets]).mean()
        # detail_score = torch.log10(detail_score + 1e-8)  # log scale for better dynamic range
        return detail_score #more negative means less details, smoother, image gradients etc

    def compute_coverage(self, img_tensor, bins = 64):
        """Computes coverage as the fraction of non-zero pixels."""
        gray_bg = torch.full_like(img_tensor, 0.5)
        overlay_img = img_tensor + ((1.0 - img_tensor) * gray_bg)
        hist_overlay = torch.histc(overlay_img, bins=bins, min=-0, max=1)
        hist_gray = torch.histc(gray_bg, bins=bins, min=0, max=1)
        #normalization over all the pixels not the bins
        hist_overlay = hist_overlay / hist_overlay.sum()
        hist_gray = hist_gray / hist_gray.sum()
        # histogram IoU
        coverage_score = 1.0 - torch.min(hist_overlay, hist_gray).sum().item()
        return coverage_score, overlay_img

    def __iter__(self):
        with tarfile.open(self.tar_path, "r|gz") as tar:
            for member in tar:
                if member.isfile() and member.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Extract class name (folder)
                    parts = member.name.split('/')
                    class_name = parts[-2] if len(parts) > 1 else "root"
                    
                    # Read image
                    f = tar.extractfile(member)
                    img = Image.open(io.BytesIO(f.read()))
                    img_tensor = self.transform(img)
                    
                    # Compute stat
                    # stat = self.compute_coverage(img_tensor)
                    deet_stat = self.compute_detail_level(img_tensor)
                    coverage_stat, _ = self.compute_coverage(img_tensor)
                    
                    yield {
                        "image": img_tensor,
                        "class": class_name,
                        "level_of_details": deet_stat,
                        "coverage": coverage_stat,
                        "name": member.name
                    }

class TextureDataModule(pl.LightningDataModule):
    def __init__(self, tar_path, batch_size=1):
        super().__init__()
        self.tar_path = tar_path
        self.batch_size = batch_size
        self.summary_stats = {}

    def prepare_data(self):
        """Analyze the tar file to get summary stats per class."""
        dataset = TextureAnalysisDataset(self.tar_path)
        class_data = {}

        print(f"--- Analyzing Dataset: {self.tar_path} ---")
        for sample in dataset:
            c = sample['class']
            if c not in class_data:
                class_data[c] = []
            class_data[c].append((sample['name'], sample['level_of_details'], sample['coverage']))

        for class_name, data in class_data.items():
            names = [d[0] for d in data]
            deet_stats = np.array([d[1] for d in data])
            coverage_stats = np.array([d[2] for d in data])
            
            # Quantitative Summary
            d_avg_val = np.mean(deet_stats)
            c_avg_val = np.mean(coverage_stats)
            min_d_idx, max_d_idx = np.argmin(deet_stats), np.argmax(deet_stats)
            min_c_idx, max_c_idx = np.argmin(coverage_stats), np.argmax(coverage_stats)
            # Find image closest to average
            avg_d_idx = np.argmin(np.abs(deet_stats - d_avg_val))
            avg_c_idx = np.argmin(np.abs(coverage_stats - c_avg_val))

            self.summary_stats[class_name] = {
                "count": len(deet_stats),
                "avg_deets": d_avg_val,
                "avg_coverage": c_avg_val,
                "min_deets": deet_stats[min_d_idx],
                "max_deets": deet_stats[max_d_idx],
                "min_coverage": coverage_stats[min_c_idx],
                "max_coverage": coverage_stats[max_c_idx],
                "names": {
                    "min_d_img": names[min_d_idx],
                    "max_d_img": names[max_d_idx],
                    "avg_d_img": names[avg_d_idx],

                    "min_c_img": names[min_c_idx],
                    "max_c_img": names[max_c_idx],
                    "avg_c_img": names[avg_c_idx],
                }
            }
            self._print_class_report(class_name)

    def _print_class_report(self, class_name):
        s = self.summary_stats[class_name]
        print(f"\nClass: {class_name}")
        print(f"  Count: {s['count']}")
        print(f"  Level of Detail Stats: Avg={s['avg_deets']:.6f} | Min={s['min_deets']:.6f} | Max={s['max_deets']:.6f}")
        print(f"  Min Detail Image: {s['names']['min_d_img']}")
        print(f"  Max Detail Image: {s['names']['max_d_img']}")
        print(f"  Avg Detail Image: {s['names']['avg_d_img']}")
        
        print(f"  Coverage Stats: Avg={s['avg_coverage']:.6f} | Min={s['min_coverage']:.6f} | Max={s['max_coverage']:.6f}")
        print(f"  Min Coverage Image: {s['names']['min_c_img']}")
        print(f"  Max Coverage Image: {s['names']['max_c_img']}")
        print(f"  Avg Coverage Image: {s['names']['avg_c_img']}")
    def train_dataloader(self):
        return DataLoader(TextureAnalysisDataset(self.tar_path), batch_size=self.batch_size)

if __name__ == '__main__':
    mode = 'train'

    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s | %(levelname)s | %(message)s"
    # )
    # data_cfg = DataConfig()
    dataset = TextureDataModule(tar_path = "../dtd-r1.0.1.tar.gz", batch_size=4).prepare_data()
    

    