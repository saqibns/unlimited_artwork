import torch
from torch import nn
import torch.nn.functional as F


class NormalFC(nn.Module):
    """
    Fully connected layer with weights are drawn from a normal distribution
    """
    def __init__(self, ins, outs):
        super().__init__()
        self.fc = nn.Linear(ins, outs)
        nn.init.normal_(self.fc.weight)

    def forward(self, x):
        return self.fc(x)


class UniformFC(nn.Module):
    """
    Fully connected layer with weights are drawn from a uniform distribution
    """
    def __init__(self, ins, outs):
        super().__init__()
        self.fc = nn.Linear(ins, outs)
        nn.init.uniform_(self.fc.weight)

    def forward(self, x):
        return self.fc(x)



class XavierNormalFC(nn.Module):
    """
    Fully connected layer with Xavier Normal initialization scheme
    """
    def __init__(self, ins, outs):
        super().__init__()
        self.fc = nn.Linear(ins, outs)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        return self.fc(x)


class XavierUniformFC(nn.Module):
    """
    Fully connected layer with Xavier Uniform initialization scheme
    """
    def __init__(self, ins, outs):
        super().__init__()
        self.fc = nn.Linear(ins, outs)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        return self.fc(x)


class KaimingNormalFC(nn.Module):
    """
    Fully connected layer with Kaiming Normal initialization scheme
    """
    def __init__(self, ins, outs):
        super().__init__()
        self.fc = nn.Linear(ins, outs)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        return self.fc(x)


class KaimingUniformFC(nn.Module):
    """
    Fully connected layer with Kaiming Uniform initialization scheme
    """
    def __init__(self, ins, outs):
        super().__init__()
        self.fc = nn.Linear(ins, outs)
        nn.init.kaiming_uniform_(self.fc.weight)

    def forward(self, x):
        return self.fc(x)


class FCWithActivation(nn.Module):
    """
    Fully connected layer with an activation function
    """
    def __init__(self, lin_class, ins, outs, activation):
        super().__init__()
        self.fc = lin_class(ins, outs)
        self.activation = activation

    def forward(self, x):
        x = self.fc(x)
        if self.activation == 'relu':
            return F.relu(x)
        if self.activation == 'sigmoid':
            return torch.sigmoid(x)
        if self.activation == 'tanh':
            return torch.tanh(x)
        if self.activation == 'softplus':
            return F.softplus(x)
