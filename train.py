# Step 5: PyTorch Lightning

import pytorch_lightning as pl
from torch import  nn 
# from models.unet_base import UNet
from models.unet_efb0 import UNet_pcb
# from models.unet_ternaus import UNet_ternaus
# from models.unet_swin import SwinUnet
from utils import *
from models.model_utils import InitWeights_XavierUniform
# import wandb
from config import *
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataset import BasicDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
# from lightning.pytorch import Trainer, seed_everything
import os


class Img_2_Img(pl.LightningModule):
    def __init__(self, model):
        super(Img_2_Img, self).__init__()
        self.model = model
        # self.model.apply(InitWeights_XavierUniform(gain=1))
        self.init_model_weights()
        # self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        # self.perceptual_loss = PerceptualLoss_reluvariant()#PercetualLoss_convvariant()
        # self.style_loss = StyleLoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.log_image_every_n_epochs = 10
        self.example_val_batch = None
        self.lr = train_cfg.lr
        

    def forward(self, x):
        return self.model(x)
    
    def init_model_weights(self):
        custom_components = [
            self.model.bottleneck, 
            self.model.decoder1, self.model.decoder2, 
            self.model.decoder3, self.model.decoder4, 
            self.model.decoder5, self.model.final_conv
        ]

        for component in custom_components:
            for m in component.modules():
                if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    # Xavier Uniform centers weights at 0 with specific variance
                    torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0.0)

        # 3. Force the Final Layer Bias to -1.0
        # This breaks the 0.50 Sigmoid floor by shifting the default output
        if hasattr(self.model.final_conv.layers[0], 'bias') and self.model.final_conv.layers[0].bias is not None:
            torch.nn.init.constant_(self.model.final_conv.layers[0].bias, -1.0)

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['mask']
        # print(images.dtype, labels.dtype)
        preds = self(images)
        # if batch_idx == 0:
        #     print(f"Preds range: {preds.min():.2f}-{preds.max():.2f} | Labels range: {labels.min():.2f}-{labels.max():.2f}")
        
        preds = preds.repeat(1, 3, 1, 1)
        labels = labels.repeat(1, 3, 1, 1)
        # print(preds.shape, labels.shape)
        
        # loss = self.MSE(preds, labels)
        loss = model_cfg.alpha_l *self.L1(preds, labels) #+ \
                # model_cfg.beta_l * self.perceptual_loss(preds, labels) + \
                # model_cfg.gamma_l * self.style_loss(preds, labels)
        psnr = self.psnr(preds, labels)
        ssim = self.ssim(preds, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log("train_psnr", psnr, prog_bar=False)
        self.log("train_ssim", ssim, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['mask']
        preds = self(images)
        # if batch_idx == 0:
        #     print(f"Preds range: {preds.min():.2f}-{preds.max():.2f} | Labels range: {labels.min():.2f}-{labels.max():.2f}")
        
        preds = preds.repeat(1, 3, 1, 1)
        labels = labels.repeat(1, 3, 1, 1) 
        # print(preds.shape, labels.shape)

        # loss = self.MSE(preds, labels)
        loss = model_cfg.alpha_l * self.L1(preds, labels)#+ \
            #    model_cfg.beta_l * self.perceptual_loss(preds, labels) + \
            #    model_cfg.gamma_l * self.style_loss(preds, labels)
        psnr = self.psnr(preds, labels)
        ssim = self.ssim(preds, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log("val_psnr", psnr, prog_bar=False)
        self.log("val_ssim", ssim, prog_bar=True)

    # def test_step(self, test_batch, batch_idx):
    #     images, labels = test_batch['image'], test_batch['mask']
    #     preds = torch.sigmoid(self.forward(images))
    #     # loss = self.MSE(preds, labels)
    #     loss = train_cfg.alpha_l * self.L1(preds, labels) + \
    #            train_cfg.beta_l * self.perceptual_loss(preds, labels) + \
    #            train_cfg.gamma_l * self.style_loss(preds, labels)
        
    #     self.log('test_loss', loss, prog_bar=True)

    # def predict_step(self, test_batch, batch_idx):
    #     images= test_batch['image']
    #     return torch.sigmoid(self.forward(images))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',              # because we monitor val_loss
        #     factor=0.5,              # lr = lr * factor
        #     patience=train_cfg.patience,              # epochs with no improvement
        #     min_lr=1e-6,
        # )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=50,       # Number of epochs until the first restart
            T_mult=2,     # Double the length of each subsequent cycle (50, 100, 200...)
            eta_min=1e-6  # Minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # MUST match self.log(...)
                "interval": "epoch",
                "frequency": 1,
            }
        }
    
    def on_validation_end(self):
        # print('Logging validation images...')
        # self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True, )
        epoch = self.current_epoch

        if (
            self.example_val_batch is None
            or epoch % self.log_image_every_n_epochs != 0
        ):
            return

        images, labels, preds = self.example_val_batch

        # Take first 8 samples
        images = images[:8].cpu()
        labels = labels[:8].cpu()
        preds  = preds[:8].cpu()

        # Your visualization function
        show_image_mask_grid(
            images,
            preds,  # or labels, depending what you want to show
            nrow=4,
            img_name=f"outputs/images/val_pred_epoch_{epoch}"
        )

        # Optional: W&B logging
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log({
                "val_examples": wandb.Image(
                    f"misc/val_pred_epoch_{epoch}.png",
                    caption=f"Epoch {epoch}"
                )
            })

        # Free memory
        self.example_val_batch = None

if __name__ == '__main__':
    # print('Training UNET model...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Using device:', device)
    print(os.cpu_count(), 'CPU cores detected.')
    
    data_cfg = DataConfig()
    # model_cfg = ModelConfig_Base()
    model_cfg = ModelConfig_ENB0()
    # model_cfg = ModelConfig_Ternaus()
    # model_cfg = ModelConfig_Swin()

    train_cfg = TrainConfig()

    train_ds = BasicDataset(f'{data_cfg.dataset_path}/train/image_v3',
                        f'{data_cfg.dataset_path}/train/mask_v2',
                        img_size=data_cfg.img_size,)
    val_ds = BasicDataset(f'{data_cfg.dataset_path}/train/image_v3',
                        f'{data_cfg.dataset_path}/train/mask_v2',
                        img_size=data_cfg.img_size,
                        split='val')
    # val_ds = train_ds
    # train_dset = Subset(train_ds, range(0, int(len(train_ds)*train_cfg.tr_split)))
    train_dset = train_ds
    train_dl = DataLoader(train_dset, batch_size=data_cfg.batch_size, shuffle=True, num_workers=data_cfg.num_workers,
                            pin_memory=True, persistent_workers=True, prefetch_factor=2)
    # val_ds = Subset(val_ds, range(int(len(train_ds)*train_cfg.tr_split), len(train_ds)))
    val_dl = DataLoader(val_ds, batch_size=data_cfg.batch_size, shuffle=False, num_workers=data_cfg.num_workers, 
                        pin_memory=True, persistent_workers=True, prefetch_factor=2)

    print(f'Train dataset size: {len(train_dset)}', 'Val dataset size:', len(val_ds))
    
    wandb_logger = WandbLogger(
        project = 'unet_i2i',
        name = model_cfg.model_name+"_1i1cm_am3",
        save_dir = 'logs',
        log_model = True,
        notes = f"Training with L1 loss, randomized masks, some training optimizations",
        # accelerator = device,
        # devices =train_cfg.num_devices,
        # precision = train_cfg.precision
    )
    

    # Log configs
    log_dataclass_config(
        wandb_logger,
        data=data_cfg,
        model=model_cfg,
        train=train_cfg,
    )

    # model = Img_2_Img().to(device)

    #SWIN
    # model = SwinUnet(config=None, img_size=data_cfg.img_size, num_classes=1)
    # model.swin_unet.load_state_dict(torch.load('pretrained_wts/swin_tiny_patch4_window7_224.pth', map_location=device)['model'], strict=False)
    
    #TERNAUS
    # model = UNet_ternaus(pretrained=model_cfg.pretrained, num_filters=model_cfg.num_filters, is_deconv=model_cfg.is_deconv)
    
    #EFB0
    model = UNet_pcb(pretrained=model_cfg.pretrained,
                     layer1_features=model_cfg.layer1_features,
                     layer2_features=model_cfg.layer2_features,
                     layer3_features=model_cfg.layer3_features,
                     layer4_features=model_cfg.layer4_features,
                     layer5_features=model_cfg.layer5_features) 

    #BASE UNET
    # model = UNet(in_channels=model_cfg.in_channels, out_channels=1)
    
    model = Img_2_Img(model).to(device)
    
    early_stop = EarlyStopping(
        monitor = train_cfg.monitor,
        patience = train_cfg.patience*2,
        mode = 'min',
        verbose = True
    )

    ckpt = ModelCheckpoint(
        monitor = train_cfg.monitor,
        save_top_k = 1,
        mode = 'min',
        filename=f"{model_cfg.model_name}"
    )

    # wandb_logger.experiment_config['num_params'] = sum(p.numel() for p  in model.parameters())

    trainer = pl.Trainer(max_epochs = train_cfg.max_epochs,
                            devices = train_cfg.num_devices,
                            precision = train_cfg.precision,
                            accumulate_grad_batches = train_cfg.accumulate_grad_batches,
                            logger = wandb_logger,
                            callbacks = [early_stop, ckpt],
                            log_every_n_steps = 10
                            )
    trainer.fit(model, train_dl, val_dl)
    # trainer.test(model, test_loader)
    # preds = trainer.predict(model, test_loader)