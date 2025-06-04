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

    all_arrays= np.empty((1, 288,len(sequences)))
    
    for i in range(len(sequences)):
        all_arrays[0,:,i]=sequences[i]['Volume'].values   
    all_arrays=torch.from_numpy(all_arrays.astype('float32')).to(device)

    CLUSTER = DF.loc[DF['cluster'] == cluster_belong]  
    know_nodes = set(CLUSTER['Unnamed: 0'])
    #test_node= set(DF.loc[DF['cluster'] == stationname]['Unnamed: 0'])
    adj = np.load('/Users/yongcanhuang/streamlit/adj_mat.npy')
    A_dynamic = adj[list(know_nodes), :][:, list(know_nodes)]   
    A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32')).to(device)
    A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32')).to(device)

    X_res = DGCN(all_arrays, A_q, A_h).data.cpu().numpy()
    
    index = other_stations_ls.index(stationname)

    return X_res[:,:,index], index


def krige_og(sequences, X_res,index):
    input_seq=sequences.copy()[index]['Volume'].values
    fig,ax = plt.subplots(figsize = (16,5))
    ax.plot(input_seq,label='Input Sequence',linewidth=2,zorder=5,color='gray')
    ax.plot(X_res[0],label='Kriging Sequence',linewidth=2,zorder=5,color='orange')
    ax.set_ylabel('CCS Counts',fontsize=20)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    plt.title('Kriging result for '+sequences[index]['Site'].iloc[0]+'_'+  sequences[index]['Direction'].iloc[0]+' '+ sequences[index]['Time'].iloc[0][0:10],fontsize=20)
   # ax.set_xticklabels(['0:00','4:00','8:00','12:00','16:00','20:00','12:00'])
    ax.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0,fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)




def error_calculate(sequences, X_res, index):
    input_seq=sequences.copy()[index]['Volume'].values
    error = input_seq-X_res[0]
    df=pd.DataFrame(error, columns=['Error'])
    df['Location']=df.index

    high=df['Error'].mean() + 2.5*df['Error'].std()
    low=df['Error'].mean() - 2.5*df['Error'].std()
    new_df = df['Error'][(df['Error']< high) & (df['Error'] < low)]

    # Define the indices of the points you want to replace
    indices_to_replace = list(new_df.index.values)  # Replace points at index 1 and 3
    # Define the new values you want to replace with
    new_values = X_res[0][indices_to_replace]
    # Update the time series array with the new values at the specified indices
    input_seq[indices_to_replace] = new_values

    fig,ax = plt.subplots(figsize = (16,5))
    ax.plot(input_seq,label='Input Sequence',linewidth=2,zorder=5,color='gray')
    #ax.plot(X_res[0],label='Kriging Sequence',linewidth=2,zorder=5,color='orange')
    #ax.plot(insert_zero(truth[288*4:288*5,3]),label= 'Faulty Timeseries', color='r',linewidth=1,zorder=0)
    plt.scatter(indices_to_replace, new_values, color='red', marker='x', label='Replaced Points',zorder=10)
    #ax.plot(o[288*4:288*5,3],label='Kriging Results',linewidth = 3,zorder=10,color='orange')
    ax.set_ylabel('CCS Counts',fontsize=20)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    plt.title('Sequence Rectification for '+sequences[index]['Site'].iloc[0]+'_'+  sequences[index]['Direction'].iloc[0]+' '+ sequences[index]['Time'].iloc[0][0:10],fontsize=20)
   # ax.set_xticklabels(['0:00','4:00','8:00','12:00','16:00','20:00','12:00'])
    ax.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0,fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    #plt.savefig('fig/metr_ignnk_temporal.pdf')
  

    return indices_to_replace


