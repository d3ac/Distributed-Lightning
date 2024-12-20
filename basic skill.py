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
        return loss

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
    dataloader = DataLoader(datasets.MNIST(root='/home/d3ac/Desktop/dataset', train=True, download=True, transform=trans), batch_size=32, shuffle=True, num_workers=7, pin_memory=True)
    dataloader = fabric.setup_dataloaders(dataloader)

    # model
    autoencoder = LitAutoEncoder(Encoder(), Decoder())
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(autoencoder, dataloader)