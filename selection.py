import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium, folium_static
import matplotlib.pyplot as plt
from folium import GeoJson
from folium.plugins import MarkerCluster
import datetime
from plot import *



def multi_option_station(DF,other_stations_ls,other_stations):

    options = st.multiselect(
        'Enter all stations in the cluster:',
        list(DF.stationname.values),max_selections=12, label_visibility='visible')

    if set(options)==set(other_stations_ls):
        
        st.write('You selected:', options)
        st.write(':green[Congrats! You have selected all correlated stations!]')

    else:
        st.warning('please select all fraternal stations:'+ other_stations )
    
    return options


def option_date(Date):
    needle=0
    d = st.date_input("Enter the date of the input sequence:", datetime.date(2019, 7, 6))
    calendar= d.timetuple()
    t = str(calendar[1])+'/'+str(calendar[2])+'/'+str(calendar[0])[2:4]#year month day
    if Date == t:
        st.write(':green[Congrats! Your date input is correct!]')
        needle=needle+1
    else:
        st.warning('please select corresponding date: '+ Date)
        needle = needle-1
    
    return d.isoformat(),needle
