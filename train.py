# Step 5: PyTorch Lightning

import pytorch_lightning as pl
from torch import  nn 
# from models.unet_base import UNet
# from models.unet_efb0 import UNet_pcb
# from models.unet_ternaus import UNet_ternaus
from models.unet_swin import SwinUnet
from utils import *
# import wandb
from config import *
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataset import BasicDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
# from lightning.pytorch import Trainer, seed_everything

class Img_2_Img(pl.LightningModule):
    def __init__(self, model):
        super(Img_2_Img, self).__init__()
        self.model = model
        self.MSE = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['mask']
        # print(images.dtype, labels.dtype)
        preds = self(images)
        
        loss = self.MSE(preds, labels)
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['mask']
        preds = self(images)

        loss = self.MSE(preds, labels)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch['image'], test_batch['mask']
        preds = self.forward(images)
        loss = self.MSE(preds, labels)
        
        self.log('test_loss', loss, prog_bar=True)

    def predict_step(self, test_batch, batch_idx):
        images, labels = test_batch['image'], test_batch['mask']
        return self.forward(images)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=train_cfg.lr)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',              # because we monitor val_loss
            factor=0.5,              # lr = lr * factor
            patience=5,              # epochs with no improvement
            min_lr=1e-6,
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

if __name__ == '__main__':
    # print('Training UNET model...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_cfg = DataConfig()
    # model_cfg = ModelConfig_Base()
    # model_cfg = ModelConfig_EFB0()
    # model_cfg = ModelConfig_Ternaus()
    model_cfg = ModelConfig_Swin()

    train_cfg = TrainConfig()

    train_ds = BasicDataset(f'{data_cfg.dataset_path}/train/masked',
                        f'{data_cfg.dataset_path}/train/mask',
                        img_size=data_cfg.img_size,)
    val_ds = BasicDataset(f'{data_cfg.dataset_path}/train/masked',
                        f'{data_cfg.dataset_path}/train/mask',
                        img_size=data_cfg.img_size,
                        split='val')
    train_dset = Subset(train_ds, range(0, int(len(train_ds)*train_cfg.tr_split)))
    train_dl = DataLoader(train_dset, batch_size=data_cfg.batch_size, shuffle=True, num_workers=data_cfg.num_workers)
    val_ds = Subset(val_ds, range(int(len(train_ds)*train_cfg.tr_split), len(train_ds)))
    val_dl = DataLoader(val_ds, batch_size=data_cfg.batch_size, shuffle=True, num_workers=data_cfg.num_workers)

    print(f'Train dataset size: {len(train_dset)}', 'Val dataset size:', len(val_ds))
    
    wandb_logger = WandbLogger(
        project = 'unet_i2i',
        name = model_cfg.model_name,
        save_dir = 'logs',
        log_model = True,
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
    model = SwinUnet(config=None, img_size=data_cfg.img_size, num_classes=1)
    model.swin_unet.load_state_dict(torch.load('pretrained_wts/swin_tiny_patch4_window7_224.pth', map_location=device)['model'], strict=False)
    model = Img_2_Img(model).to(device)
    
    early_stop = EarlyStopping(
        monitor = train_cfg.monitor,
        patience = train_cfg.patience,
        mode = 'min',
        verbose = True
    )

    ckpt = ModelCheckpoint(
        monitor = train_cfg.monitor,
        save_top_k = 1,
        mode = 'min',
        filename="ckpts/{model_config.model_name}/epoch{epoch}-val{val_loss:.4f}"
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