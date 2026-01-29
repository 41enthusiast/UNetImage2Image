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
from albumentations.pytorch import ToTensorV2
import albumentations as A
# from utils import show_image_mask_grid
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
        masks_vis = (masks_vis / 255).clamp(min=1)

    # Convert masks to 3-channel so grid looks consistent
    # masks_vis = masks_vis.repeat(1, 3, 1, 1)

    # Stack image above mask for each sample
    stacked = torch.cat([images, masks_vis], dim=2)
    # stacked shape: (B, 3, 2H, W)
    from torchvision import utils as vutils
    import matplotlib.pyplot as plt
    grid = vutils.make_grid(
        stacked,
        nrow=nrow,
        padding=2,
        normalize=True,
        # value_range=(-1, 1)  # remove if images already [0,1]
    )

    plt.figure(figsize=(12, 6))
    plt.imsave(f'{img_name}.png',grid.permute(1, 2, 0),)


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, img_size=None, mask_suffix: str = '', split = 'train',
                 n_images = None, n_subsets: int = 50, class_names: list = None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.img_size = [img_size, img_size]
        self.mask_suffix = mask_suffix
        self.class_names = class_names
        self.n_images = n_images
        self.n_subsets = n_subsets

        #to mod next to have a certain cutoff of train datasets
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        
        # self.mask_ids = [splitext(file)[0] for file in listdir(mask_dir) if isfile(join(mask_dir, file)) and not file.startswith('.')]
        # self.mask_ids = self.mask_ids*(len(self.ids)//len(self.mask_ids)+1)[:len(self.ids)] #repeat mask ids if less masks than images
        self.texture_masks = {}
        for cls in self.class_names:
            cls_path = self.mask_dir / cls
            if cls_path.is_dir():
                self.texture_masks[cls] = sorted(list(cls_path.glob('*.*'))) #currently doesnt check for hidden files
            else:
                print(f'Warning: Class folder {cls_path} does not exist in mask directory.')

        if split == 'train':
            self.total_units = len(self.class_names) * self.n_subsets
        else:
            self.total_units = sum([len(self.texture_masks[cls]) for cls in self.class_names])

        self.shuffled_ids = list(range(len(self.class_names))) #check back to see if it samples images from the classes

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

        

        self.transform = self.data_tfm(self.img_size)
        self.val_transform = self.val_tfm()
        self.split = split

        # print('New image size:', self.img_size)

    def __len__(self):
        if self.split == 'train':
            return len(self.ids) * self.total_units
        else:
            return len(self.ids)

    @staticmethod
    def val_tfm():
            return A.Compose([
                # A.Resize(height=img_size[0], width=img_size[1], interpolation=4),
                # A.Normalize(mean=(0.,), std=(1.0,)),
                ToTensorV2()
            ])

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
                # A.Resize(height=img_size[0], width=img_size[1], interpolation=4),
                #Resize Crop
                # A.RandomResizedCrop(
                #     size=img_size,
                #     scale=(resize_range_min, 1.0),
                #     ratio=(1.0, 1.0),#dont change the aspect ratio
                #     p=1.0,
                #     interpolation=4
                # ),
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
            img = img.repeat(3, axis=2)
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
        img_idx = idx // self.total_units # because getitem is called len(ids)*total_units times
        name = self.ids[img_idx]
        # mask_file = list(self.mask_dir.glob('*.*'))
        # random.shuffle(mask_file)
        # mask_file = mask_file[idx % len(mask_file)]
        img_file = list(self.images_dir.glob(name + '.*'))#image folder

        if self.split == 'train':
            mask_idx = idx % self.total_units
            cls_idx = mask_idx %len(self.class_names)
            subset_idx = mask_idx // len(self.class_names)
            tgt_cls = self.class_names[self.shuffled_ids[cls_idx]]
            available_masks = self.texture_masks[tgt_cls]
            mask_file = available_masks[subset_idx % len(available_masks)]
        else:
            cls_idx = img_idx % len(self.class_names)#to represent the classes per image
            cls_name = self.class_names[cls_idx]
            available_masks = self.texture_masks[cls_name]
            mask_idx = img_idx % len(available_masks)
            mask_file = available_masks[mask_idx]
            

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        # assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file, mode='L')
        img = load_image(img_file[0])
        masked = self.overlay_mask_on_image(img, mask)

        img = self.preprocess(masked, self.scale, self.img_size)
        mask = self.preprocess(mask, self.scale, self.img_size)
        
        if self.split == 'train':
            # print('input ', img.shape, mask.shape)
            augmented = self.transform(image = img, mask=mask)
            # print('augmented', augmented['image'].shape, augmented['mask'].shape)
        else:
            # print(img.shape, mask.shape)
            augmented = {'image': torch.as_tensor(img),
                         'mask': torch.as_tensor(mask)
                        }
            augmented = self.val_transform(image = img, mask=mask)
        
        # assert augmented['image'].size() == augmented['mask'].size(), \
        #     f"Image {name} and mask should be the same size, but are {augmented['image'].size()} and {augmented['mask'].size()}"

        return {
            'image': augmented['image'].float().contiguous(),
            'mask':augmented['mask'].permute(2,0,1).float().contiguous()
        }

if __name__ == '__main__':
    mode = 'train'

    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s | %(levelname)s | %(message)s"
    # )
    from config import DataConfig
    data_cfg = DataConfig()
    train_ds = BasicDataset(f'{data_cfg.dataset_path}/train/image',
                        data_cfg.mask_path,
                        img_size=data_cfg.img_size,
                        split='train',
                        class_names = data_cfg.class_names,
                        n_subsets=data_cfg.n_subsets)
    val_ds = BasicDataset(f'{data_cfg.dataset_path}/train/image',
                        data_cfg.mask_path,
                        img_size=data_cfg.img_size,
                        split='val',
                        class_names = data_cfg.class_names,
                        n_subsets=data_cfg.n_subsets)
    print(f'Train dataset size: {len(train_ds)}', 'Val dataset size:', len(val_ds))
    print(train_ds.total_units, val_ds.total_units)
    train_dl = DataLoader(train_ds, batch_size=data_cfg.batch_size, shuffle=True, num_workers=data_cfg.num_workers,
                        pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_dl = DataLoader(val_ds, batch_size=data_cfg.batch_size, shuffle=False, num_workers=data_cfg.num_workers,
                        pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_counts = 0
    for batch in val_dl:
        val_counts += batch['image'].shape[0]
    print('Val counts:', val_counts)
    print(batch['image'].shape, batch['mask'].shape)
    show_image_mask_grid(batch['image'], batch['mask'], nrow=4, img_name='../outputs/images/val_imgmask_grid')
    
    counts = 0
    for batch in train_dl:
        counts += batch['image'].shape[0]
    print(counts, batch['image'].shape, batch['mask'].shape)
    show_image_mask_grid(batch['image'], batch['mask'], nrow=4, img_name='../outputs/images/train_imgmask_grid')
    
    # dataset = TextureDataModule(tar_path = "../dtd-r1.0.1.tar.gz", batch_size=4).prepare_data()
    

    