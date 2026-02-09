
import pytorch_lightning as pl
from torch import  nn 
# from models.unet_base import UNet
from models.unet_efb0 import UNet_pcb
# from models.unet_ternaus import UNet_ternaus
# from models.unet_swin import SwinUnet
from utils import *
# import wandb
from config import *
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
# from lightning.pytorch import Trainer, seed_everything
import os
import webdataset as wds
from dataset.dataset import ArtDatasetPipeline, apply_transforms, get_pipeline

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MASK_DIR = '../art_painting_data/new_test/mask'
IMG_DIR = '../art_painting_data/new_test/image'
# DATA_ROOT = ''
class ArtDataModule(pl.LightningDataModule):
    def __init__(self, test_pipeline, batch_size=16, num_workers=4):
        super().__init__()
        self.test_pipeline = test_pipeline
        self.batch_size = batch_size
        self.num_workers = num_workers

    def test_dataloader(self):
        # We use wds.WebLoader because it's optimized for WebDataset pipelines
        return wds.WebLoader(
            self.test_pipeline, 
            batch_size=None, # Batching is already handled in the pipeline
            num_workers=self.num_workers,
        )

class Img_2_Img(pl.LightningModule):
    def __init__(self, model):
        super(Img_2_Img, self).__init__()
        self.model = model
        self.model.eval()
        self.L1 = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss_reluvariant()#PercetualLoss_convvariant()
        self.style_loss = StyleLoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    
    def test_step(self, batch, batch_idx):
        images, labels = batch['masked_image'], batch['mask_image']
        recon = batch['recon_masked_image']
        preds = self.model(images)

        loss = self.L1(preds, labels)
        err = self.L1(images, recon)
        psnr = self.psnr(preds, labels)
        ssim = self.ssim(preds, labels)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_psnr', psnr)
        self.log('test_ssim', ssim)
        self.log('masked_recon_loss', err)

        return {'loss': loss, 'preds': preds, 'target': labels, 'recon_err': err}


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    CKPT = "ckpts/unet_efb0_allc50s.ckpt"

    print('Using device:', device)
    print(os.cpu_count(), 'CPU cores detected.')

    model_cfg = ModelConfig_ENB0()

    wandb_logger = WandbLogger(
        project = 'unet_i2i',
        name = model_cfg.model_name+"_v3_ad_dtdALLc_50s_inference",
        save_dir = 'logs',
        log_model = True,
        notes = f"Testing with 3 levels of alpha split datasets",
    )

    # Creating the Pipeline
    test_dataset = wds.DataPipeline(
                    get_pipeline,
                    wds.map(apply_transforms),
                    wds.batched(8)
                )
    art_alpha_module = ArtDataModule(test_dataset)
    model = UNet_pcb(pretrained=model_cfg.pretrained,
                     layer1_features=model_cfg.layer1_features,
                     layer2_features=model_cfg.layer2_features,
                     layer3_features=model_cfg.layer3_features,
                     layer4_features=model_cfg.layer4_features,
                     layer5_features=model_cfg.layer5_features) 
    model = Img_2_Img.load_from_checkpoint(model = model,
                                           checkpoint_path = CKPT)

    # 2. Setup Trainer
    trainer = pl.Trainer(
        accelerator=device,
        devices=1,
        logger=wandb_logger # Optional
    )
    trainer.test(model=model, datamodule=art_alpha_module)

