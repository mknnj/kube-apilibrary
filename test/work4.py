import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin


class TestNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v2()

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x_hat = self.model(x)
        loss = F.nll_loss(x_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def test_ext():
    dataset = CIFAR10("/datasets/cifar", download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [45000, 5000])
    num_nodes = int(os.getenv("NUM_REPLICAS"))
    batch_size = int(os.getenv("BATCH_SIZE"))
    workers = int(os.getenv("NUM_WORKERS"))
    epochs = int(os.getenv("NUM_EPOCHS"))
    print("device count from inside pod: "+str(torch.cuda.device_count()))
    elastic_percent = int(os.getenv("ELASTIC_PERCENT"))
    full_gpus = os.getenv("ELASTIC_FULL_GPUS")
    num_dev = 1
    if full_gpus == "TRUE":
       num_dev = -1
    else:
       torch.cuda.set_per_process_memory_fraction(elastic_percent/2000)
       print("Percent: "+str(elastic_percent/2000))
    restarted = os.getenv("JOB_RESTARTED")
    uuid = os.getenv("JOB_UUID")
    dirpath = os.path.join("/snapshots", uuid)
    if restarted == "FALSE" and not os.path.isdir(dirpath):
       os.mkdir(dirpath)
    nn = TestNN()
    trainer = pl.Trainer(accelerator="gpu", devices = num_dev, num_nodes=num_nodes, strategy=DDPPlugin(find_unused_parameters=False), max_epochs=epochs, profiler="simple", default_root_dir=dirpath)
    if restarted == "FALSE":
       trainer.fit(nn, DataLoader(train, batch_size=batch_size, num_workers= workers, persistent_workers=True, drop_last=True))
    else:
       checkpoints_dir = os.path.join(dirpath, "lightning_logs/version_0/checkpoints")
       _, _, filenames = next(os.walk(checkpoints_dir), (None, None, []))
       trainer.fit(nn, DataLoader(train, batch_size=batch_size, num_workers= workers, persistent_workers=True, drop_last=True), ckpt_path=os.path.join(checkpoints_dir, filenames[0]))
