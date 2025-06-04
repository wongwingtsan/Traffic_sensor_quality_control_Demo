from __future__ import division
import torch.optim as optim
from utils import *
from basic_structure import D_GCN, C_GCN, K_GCN,IGNNK
import geopandas as gp
import matplotlib as mlt

from warnings import simplefilter 
simplefilter(action='ignore', category=DeprecationWarning)
import os
from PIL import Image
import gc 
import pandas as pd
from datetime import date
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
import scipy.sparse as sp
from sklearn.preprocessing import minmax_scale
import os

import math
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """       
    def __init__(self, in_channels, out_channels, orders, activation = 'relu'): 
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(D_GCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices,
                                             out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)
        
    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)
        
    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size = X.shape[0]  # batch_size
        num_node = X.shape[1]    # number of nodes
        input_size = X.size(2)   # time_length
        
        # Debug prints
        print("Input X shape:", X.shape)
        print("A_q shape:", A_q.shape)
        print("A_h shape:", A_h.shape)
        
        supports = []
        supports.append(A_q)
        supports.append(A_h)
        
        # Reshape X to (num_nodes, timesteps * batch_size)
        x0 = X.permute(1, 2, 0).reshape(num_node, -1)
        print("x0 shape after reshape:", x0.shape)
        
        x = torch.unsqueeze(x0, 0)
        print("x shape after unsqueeze:", x.shape)
        
        for support in supports:
            # Verify support matrix matches number of nodes
            if support.shape != (num_node, num_node):
                raise ValueError(f"Support matrix shape {support.shape} must match number of nodes {num_node}")
            
            x1 = torch.mm(support, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.mm(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1
                
        x = torch.reshape(x, shape=[self.num_matrices, num_node, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size, num_node, input_size * self.num_matrices])         
        x = torch.matmul(x, self.Theta1)  # (batch_size, num_nodes, output_size)     
        x += self.bias
        
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)   
            
        return x



def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()

class DGCN(nn.Module):
    """
    GNN on ST datasets to reconstruct the datasets
    x_s
    |GNN_3
    H_2 + H_1
    |GNN_2
    H_1
    |GNN_1
    x^y_m     
    """
    def __init__(self, h, z, k): 
        super(DGCN, self).__init__()
        self.time_dimension = h  # Number of time steps (288)
        self.hidden_dimension = z  # Hidden dimension size
        self.order = k  # Order of the graph convolution

        self.GNN1 = D_GCN(self.time_dimension, self.hidden_dimension, self.order)
        self.GNN2 = D_GCN(self.hidden_dimension, self.hidden_dimension, self.order)
        self.GNN3 = D_GCN(self.hidden_dimension, self.time_dimension, self.order, activation='linear')

    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_nodes, num_timesteps)
        """  
        # Input X is already in the correct shape (batch_size, num_nodes, timesteps)
        print("DGCN input shape:", X.shape)
        print("A_q shape:", A_q.shape)
        print("A_h shape:", A_h.shape)
        
        X_s1 = self.GNN1(X, A_q, A_h)
        X_s2 = self.GNN2(X_s1, A_q, A_h) + X_s1
        X_s3 = self.GNN3(X_s2, A_q, A_h)
        
        return X_s3