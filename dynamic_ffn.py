import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNet(nn.Module):

    def __init__(self, input_size=2048, output_size=4, divide_factor=3,
                 max_layers=3):
        super(FFNet, self).__init__()
        self.n_in = input_size
        self.n_out = output_size
        self.fcs = []
        while len(self.fcs) < max_layers*2:
            output_size = int(input_size/divide_factor)
            layer = nn.Linear(input_size, output_size)
            relu = nn.ReLU()
            input_size = output_size
            setattr(self, ('hidden_'+str(len(self.fcs))), layer)
            setattr(self, ('relu'+str(len(self.fcs))), relu)
            self.fcs.append(layer)
            self.fcs.append(relu)
        self.final = nn.Linear(input_size, self.n_out)
        self.fcs.append(self.final)

    def forward(self, x):
        for fc in self.fcs:
            x = fc(x)
        return x

    def loss(self):
        self
