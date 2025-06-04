import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium, folium_static
import matplotlib.pyplot as plt
from folium import GeoJson
from folium.plugins import MarkerCluster


def map_generator(text,DF):

    my_map= folium.Map(location=[DF.lat.mean(), DF.lon.mean()],zoom_start=7, control_scale=True)
    GeoJson(text).add_to(my_map)



    cluster_num = list(set(DF.cluster))
    clustername = []
    for num in cluster_num:
        clustername.append('cluster'+str(num))

    markercluster_list=[]
    for each in clustername:
        markercluster_list.append(MarkerCluster(name=each).add_to(my_map))

    #画marker
    for i, row in DF.iterrows():
        
        iframe = folium.IFrame('Station Name:'+ str(row['stationname'])+', '+'Cluster ID:'+ str(row['cluster']))
        popup = folium.Popup(iframe, min_width = 300, max_width = 300)

        folium.Marker(location = [row['lat'],row['lon']],popup=popup,c=row['stationname']).add_to(markercluster_list[row['cluster']])

    folium.LayerControl().add_to(my_map)
    st_data = folium_static(my_map,width=1300,height=600)

    return markercluster_list