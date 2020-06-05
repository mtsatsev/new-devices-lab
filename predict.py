import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import torchvision.transforms as transforms
from sklearn import train_test_split




class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d()
        self.l1 = nn.Linear()


    def predict(image):
