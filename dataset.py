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

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


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
                A.RandomResizedCrop(
                    size=img_size,
                    scale=(resize_range_min, 1.0),
                    ratio=(1.0, 1.0),#dont change the aspect ratio
                    p=1.0,
                    interpolation=4
                ),
                #Normalize - disabling for now
                # A.Normalize(mean=(0.5,), std=(0.5,)),
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


        # if is_mask:
        #     mask = np.zeros((newH, newW), dtype=np.int64)
        #     for i, v in enumerate(mask_values):
        #         if img.ndim == 2:
        #             mask[img == v] = i
        #         else:
        #             mask[(img == v).all(-1)] = i

        #     return mask

        # else:
        if img.ndim == 2:
            img = img[..., np.newaxis]
        # else:
        #     img = img.transpose((2, 0, 1))

        if (img > 1).any():
            img = img / 255.0

        return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))#masked folder

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, self.img_size)
        mask = self.preprocess(mask, self.scale, self.img_size)

        if self.split == 'train':
            augmented = self.transform(image = img, mask=mask)
        else:
            augmented = {'image': torch.as_tensor(img).permute(2,0,1),
                         'mask': torch.as_tensor(mask)
                        }
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
    data_cfg = DataConfig()
    

    train_ds = BasicDataset(f'{data_cfg.dataset_path}/{mode}/masked',
                            f'{data_cfg.dataset_path}/{mode}/mask',
                            img_size=data_cfg.img_size,
                            )
    train_dl = DataLoader(train_ds, batch_size=data_cfg.batch_size, shuffle=True, num_workers=data_cfg.num_workers)
    # print(train_ds[0]['image'].shape, train_ds[0]['mask'].shape)
    for batch in train_dl:
        print(batch['image'].dtype, batch['mask'].dtype)
        print(batch['image'].shape, batch['mask'].shape)
        show_image_mask_grid(batch['image'], batch['mask'])
        break