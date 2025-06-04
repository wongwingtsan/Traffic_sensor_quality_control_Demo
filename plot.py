import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium, folium_static
import matplotlib.pyplot as plt
from folium import GeoJson
from folium.plugins import MarkerCluster
from selection import *
from mothermap import *
import os
import datetime


###show cluster-level timeseries
def show_plot(path,station_options, date_option,stationname):
    DB = pd.read_csv(path)
    sequences=[]
    for station in station_options:
        que = DB[(DB['Time'].str[0:10]==date_option) & (DB['Site']==station[0:8]) & (DB['Direction']== station[9])].reset_index(drop=True)
        sequences.append(que)
    #plot
    fig,ax = plt.subplots(figsize = (16,5))
    for seq in sequences:
        if seq['Site'].iloc[0]+'_'+ seq['Direction'].iloc[0]==stationname:
            print(stationname)
            print(seq['Site'].iloc[0])
            ax.plot(seq['Volume'].values,label='Input Sequence',linewidth=2,color='gray')
        else:
            ax.plot(seq['Volume'].values,label=seq['Site'].iloc[0]+'_'+ seq['Direction'].iloc[0]+' '+seq['Time'].iloc[0][0:10],linewidth=1,color='yellow')

        
       # ax.plot(seq['Volume'].values,label=seq['Site'].iloc[0]+'_'+ seq['Direction'].iloc[0]+' '+seq['Time'].iloc[0][0:10],linewidth=2)

    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel('CCS Counts',fontsize=20)
   # ax.set_xticklabels(['0:00','4:00','8:00','12:00','16:00','20:00','12:00'])
    ax.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0,fontsize=12)
    plt.title('All sequences in the cluster',fontsize=20)
    plt.tight_layout()
    st.pyplot(fig)

    return sequences

def show_plot_1(sequences,stationname):

    #plot
   
    fig,ax = plt.subplots(figsize = (16,5))
    for seq in sequences:
        if seq['station'].iloc[0]==stationname:
            ax.plot(seq.iloc[0][4:288+4].values,label='Input Sequence',linewidth=2,color='gray')
        else:
            ax.plot(seq.iloc[0][4:288+4].values,label=seq['station'].iloc[0],linewidth=1,color='green')



        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel('CCS Counts',fontsize=20)
    # ax.set_xticklabels(['0:00','4:00','8:00','12:00','16:00','20:00','12:00'])
        ax.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0,fontsize=12)
        plt.title('All sequences in the cluster',fontsize=20)
        plt.tight_layout()
        st.pyplot(fig)

    return sequences