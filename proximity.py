
import torch.optim as optim
from utils import *
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



def generateDF(path_distance,path_coordinates,namelist):
    df = pd.read_csv(path_distance) 
    names1=list(df['Y0'])
    df=df.drop(columns=['Y0'])
    df.columns=names1
    CO=pd.read_csv(path_coordinates)
    df['stationname']=CO.ID
    df=df.set_index('stationname')
    df=df.loc[namelist]
    df=df[namelist]
    
    return df


def generate_adj(stationname,DF):

    matrix=pd.DataFrame(columns=stationname,index=stationname)


    for station in stationname:
        for STATION in stationname:
            matrix.loc[station,STATION]=DF.loc[station[0:8],STATION[0:8]]

    MAX=matrix.max()
    matrix =matrix/MAX
    matrix = np.exp(-matrix.astype(float))
    matrix=matrix.apply(pd.Series.nlargest, axis=1, n=493)
    matrix=matrix.fillna(0)
    matrix=matrix.values

    adj = np.zeros_like(matrix)
    for i in range(len(matrix)):
        ind=np.argsort(matrix[i])[::-1][0:6]
        adj[i][ind]=1

    print('The adjacent matrix shape is', adj.shape )
    
    return adj,matrix