from kubernetesdlprofile import kubeprofiler
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models


def test_measure():
    p = kubeprofiler.KubeProfiler(3)
    p.measure()
    p.measure()
    p.measure()
    p.measure()
    p.measure()
    p.measure()


