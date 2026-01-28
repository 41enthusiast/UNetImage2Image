
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
from dataset import BasicDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
# from lightning.pytorch import Trainer, seed_everything


class Img_2_Img(pl.LightningModule):
    def __init__(self, model):
        super(Img_2_Img, self).__init__()
        self.model = model
        self.MSE = nn.MSELoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.log_image_every_n_epochs = 10
        self.example_val_batch = None

    def forward(self, x):
        return self.model(x)
    
    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['mask']
        preds = self(images)

        loss = self.MSE(preds, labels)
        psnr = self.psnr(preds, labels)
        ssim = self.ssim(preds, labels)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_psnr', psnr)
        self.log('test_ssim', ssim)

        # ---------- Save & log images ----------
        model_name = self.hparams.model_name
        run_name = self.logger.experiment.name

        save_dir = os.path.join(
            "logs", model_name, run_name
        )
        os.makedirs(save_dir, exist_ok=True)

        # save first 4 images of batch
        for i in range(min(4, images.size(0))):
            grid = torch.cat([
                images[i],
                preds[i],
                labels[i]
            ], dim=-1)  # side-by-side

            path = os.path.join(
                save_dir, f"test_batch{batch_idx:04d}_img{i}.png"
            )

            torchvision.utils.save_image(grid, path)

            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({
                    "test_examples": wandb.Image(
                        path,
                        caption=f"batch {batch_idx} img {i}"
                    )
                })
