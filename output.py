from __future__ import division
import streamlit as st
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
from model import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_processing(z,K,h,model_path,DGCN,sequences,cluster_belong,DF,stationname,other_stations_ls,file_df):
    X_res, position= kriging_result(DGCN,sequences,cluster_belong,DF,stationname,other_stations_ls)
    krige_og(sequences, X_res,position)
    error_places=error_calculate(sequences, X_res, position)
    with st.expander('The adjusted locations in the 288 point sequence : '):
        st.write(error_places)
    
    file_df.iloc[0][4:4+288]=X_res[0]

    def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(file_df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='revised_sequence.csv',
        mime='text/csv',
    )
    
 #
    #X_res[0]-sequences[position]['Volume'].values





def kriging_result(DGCN,sequences,cluster_belong,DF,stationname,other_stations_ls):
    all_arrays = np.empty((1, 288, len(sequences)))
    
    for i in range(len(sequences)):
        # Extract the time series values (columns 4 to 291 contain the 288 time points)
        all_arrays[0,:,i] = sequences[i].iloc[0][4:292].values
    all_arrays = torch.from_numpy(all_arrays.astype('float32')).to(device)

    CLUSTER = DF.loc[DF['cluster'] == cluster_belong]  
    know_nodes = set(CLUSTER['Unnamed: 0'])
    adj = np.load('./data/adj_mat.npy')
    A_dynamic = adj[list(know_nodes), :][:, list(know_nodes)]   
    A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32')).to(device)
    A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32')).to(device)

    X_res = DGCN(all_arrays, A_q, A_h).data.cpu().numpy()
    
    index = other_stations_ls.index(stationname)

    return X_res[:,:,index], index


def krige_og(sequences, X_res, index):
    # Extract the time series values (columns 4 to 291 contain the 288 time points)
    input_seq = sequences[index].iloc[0][4:292].values
    fig,ax = plt.subplots(figsize = (16,5))
    ax.plot(input_seq, label='Input Sequence', linewidth=2, zorder=5, color='gray')
    ax.plot(X_res[0], label='Kriging Sequence', linewidth=2, zorder=5, color='orange')
    ax.set_ylabel('CCS Counts',fontsize=20)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    plt.title('Kriging result for ' + sequences[index]['station'].iloc[0] + ' ' + sequences[index]['Date'].iloc[0], fontsize=20)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)




def error_calculate(sequences, X_res, index):
    # Extract the time series values (columns 4 to 291 contain the 288 time points)
    input_seq = sequences[index].iloc[0][4:292].values
    error = input_seq - X_res[0]
    df = pd.DataFrame(error, columns=['Error'])
    df['Location'] = df.index

    high = df['Error'].mean() + 2.5*df['Error'].std()
    low = df['Error'].mean() - 2.5*df['Error'].std()
    new_df = df['Error'][(df['Error'] < high) & (df['Error'] < low)]

    indices_to_replace = list(new_df.index.values)
    new_values = X_res[0][indices_to_replace]
    input_seq[indices_to_replace] = new_values

    fig,ax = plt.subplots(figsize = (16,5))
    ax.plot(input_seq, label='Input Sequence', linewidth=2, zorder=5, color='gray')
    plt.scatter(indices_to_replace, new_values, color='red', marker='x', label='Replaced Points', zorder=10)
    ax.set_ylabel('CCS Counts', fontsize=20)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    plt.title('Sequence Rectification for ' + sequences[index]['station'].iloc[0] + ' ' + sequences[index]['Date'].iloc[0], fontsize=20)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    return indices_to_replace


