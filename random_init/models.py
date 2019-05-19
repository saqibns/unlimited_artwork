import torch
from torch import nn
import random_init.layers as layers


class HomogeneousMLP(nn.Module):
    def __init__(self, input_dim, output_dim, layer, hidden_dims, activation):
        super().__init__()
        _layers = list()
        # Create the input layer
        _layers.append(layers.FCWithActivation(layer, input_dim, hidden_dims[0], activation))

        # Create all hidden layers
        for i in range(len(hidden_dims) - 1):
            _layers.append(layers.FCWithActivation(layer, hidden_dims[i], hidden_dims[i+1], activation))

        # Create the output layer
        _layers.append(layers.FCWithActivation(layer, hidden_dims[-1], output_dim, activation))

        self.net = nn.Sequential(*_layers)

    def forward(self, x):
        x = self.net(x)
        return torch.sigmoid(x)
