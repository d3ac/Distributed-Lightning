import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import lightning as L

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(32 * 32, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 32 * 32))

    def forward(self, x):
        return self.l1(x)
    
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, prog_bar=True) # 可以在进度条中看到train_loss
        return loss
    
    def valid_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("valid_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == '__main__':
    # Fabric accelerates
    fabric = L.Fabric()
    fabric.launch()

    # load data
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataloader = DataLoader(datasets.MNIST(root='/home/d3ac/Desktop/dataset', train=True, download=True, transform=trans), batch_size=32, shuffle=True, num_workers=7, pin_memory=True)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    valid_dataloader = DataLoader(datasets.MNIST(root='/home/d3ac/Desktop/dataset', train=False, download=True, transform=trans), batch_size=32, shuffle=True, num_workers=7, pin_memory=True)
    valid_dataloader = fabric.setup_dataloaders(valid_dataloader)

    # train
    autoencoder = LitAutoEncoder(Encoder(), Decoder())
    trainer = L.Trainer(max_epochs=100)
    # Trainer(accelerator="gpu", devices=[0, 1, 2]) # device=-1就是所有设备
    trainer.fit(autoencoder, train_dataloader, valid_dataloader)

"""
启动 tensorboard : tensorboard --logdir=lightning_logs/
从检查点恢复 : model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")
batch_norm在多个GPU上是不同步的 : L.Trainer(sync_batchnorm=True)

"""