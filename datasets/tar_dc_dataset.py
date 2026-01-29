
import ptwt #pywt
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from PIL import Image
import torch
import tarfile
from torchvision import transforms
import pytorch_lightning as pl
import io

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
