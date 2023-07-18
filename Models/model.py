import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer, clip=None, dropout=None):
        super(NN, self).__init__()
        self.n_hid = n_layer - 2
        self.in_layer = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        nn.init.kaiming_uniform_(self.in_layer.weight, nonlinearity="relu")
        self.hid_layer = []
        for i in range(self.n_hid):
            layer = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            self.hid_layer.append(layer)
        self.out_layer = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        if clip is not None: nn.init.uniform_(self.out_layer.weight, a=0, b=clip)


    def forward(self, x, mode='normal'):
        h = torch.nn.functional.relu(self.in_layer(x))
        for i in range(self.n_hid):
            h = self.dropout(h) if self.dropout is not None else h
            h = torch.nn.functional.relu(self.hid_layer[i](h))
        if mode == 'normal':
            h = torch.nn.functional.sigmoid(self.out_layer(h))
            return h
        else:
            x_in = h
            x_out = self.out_layer(x_in)
            return x_in, x_out


class NormNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer, clip=None):
        super(NormNN, self).__init__()
        self.n_hid = n_layer - 2
        self.in_layer = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.in_layer.weight, nonlinearity="relu")
        self.hid_layer = []
        for i in range(self.n_hid):
            layer = nn.Linear(hidden_dim, hidden_dim)
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            self.hid_layer.append(layer)
        self.out_layer = nn.Linear(hidden_dim, output_dim)
        if clip is not None: nn.init.uniform_(self.out_layer.weight, a=0, b=clip)

    def forward(self, x, mode = 'normal'):
        h = torch.nn.functional.relu(self.in_layer(x))
        for i in range(self.n_hid):
            h = torch.nn.functional.relu(self.hid_layer[i](h))
        norm = torch.norm(h, p=1, dim=-1, keepdim=True).repeat(1, h.size(dim=-1)) + 1e-16
        h = torch.div(h, norm)
        if mode == 'normal':
            h = torch.nn.functional.sigmoid(self.out_layer(h))
            return h
        else:
            x_in = h
            x_out = self.out_layer(x_in)
            return x_in, x_out

class NormLR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NormLR, self).__init__()
        self.out_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x, mode='normal'):
        norm = torch.norm(x, p=1, dim=-1, keepdim=True).repeat(1, x.size(dim=-1)) + 1e-16
        h = torch.div(x, norm)
        if mode == 'normal':
            h = torch.nn.functional.sigmoid(self.out_layer(h))
            return h
        else:
            x_in = h
            x_out = self.out_layer(x_in)
            return x_in, x_out

class LR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LR, self).__init__()
        self.out_layer = nn.Linear(input_dim, output_dim)
        nn.init.kaiming_uniform_(self.out_layer.weight, nonlinearity="sigmoid")

    def forward(self, x, mode='normal'):
        if mode == 'normal':
            h = torch.nn.functional.sigmoid(self.out_layer(x))
            return h
        else:
            x_in = x
            x_out = self.out_layer(x_in)
            return x_in, x_out

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(1296, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 84)
        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

def init_model(args):
    if args.model_type == 'NormNN':
        if args.mode not in ['lipr_dpsgd_sum', 'lipr_dpsgd_avg', 'lipr_fairdp_sum', 'lipr_fairdp_avg', 'fairdp_reg']:
            return NormNN(input_dim=args.input_dim, hidden_dim=args.n_hid, output_dim=args.output_dim,
                          n_layer=args.n_layer)
        else:
            return NormNN(input_dim=args.input_dim, hidden_dim=args.n_hid, output_dim=args.output_dim,
                          n_layer=args.n_layer, clip=args.clip_layer)
    elif args.model_type == 'NN':
        if args.mode not in ['lipr_dpsgd_sum', 'lipr_dpsgd_avg', 'lipr_fairdp_sum', 'lipr_fairdp_avg', 'fairdp_reg']:
            return NN(input_dim=args.input_dim, hidden_dim=args.n_hid, output_dim=args.output_dim,
                      n_layer=args.n_layer, dropout=None)
        else:
            return NN(input_dim=args.input_dim, hidden_dim=args.n_hid, output_dim=args.output_dim,
                      n_layer=args.n_layer, clip=args.clip_layer, dropout=None)
    elif args.model_type == 'NormLR':
        return NormLR(args.input_dim, args.output_dim)
    elif args.model_type == 'LR':
        return LR(args.input_dim, args.output_dim)
    elif args.model_type == 'CNN':
        return CNN(args.input_dim, args.n_hid, args.output_dim)
