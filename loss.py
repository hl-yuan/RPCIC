import math
import sys

import torch
import torch.nn as nn

from utils import get_Similarity



class Cross_inscl_loss(nn.Module):

    def __init__(self):
        super(Cross_inscl_loss, self).__init__()
    def forward(self, h0, h1):

        h0, h1 = nn.functional.normalize(h0, dim=1), nn.functional.normalize(h1, dim=1)
        tao = 1
        cos = get_Similarity(h0, h1)
        sim = (cos / tao).exp()
        pos = sim.diag()
        p = pos / sim.sum(1)
        loss = -(p*torch.log(p)).mean()
        return loss

class Noise_robust_loss(nn.Module):

    def __init__(self):
        super(Noise_robust_loss, self).__init__()
    def forward(self,h0, h1 ,r):
        h0, h1 = nn.functional.normalize(h0, dim=1), nn.functional.normalize(h1, dim=1)
        tao = 1
        cos = get_Similarity(h0, h1)
        sim = (cos/tao).exp()
        pos = sim.diag()
        p = pos / sim.sum(1)
        robust_loss = (((1-p)**r) * ((-(torch.log(p)))**(1-r))).mean()
        return robust_loss
