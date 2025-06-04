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
from plot import *
from model import *
from PIL import Image
#export KMP_DUPLICATE_LIB_OK=True
import datetime
import os
from output import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import datetime


st.set_page_config('Cluster-informed CCS Data Quality Control', page_icon=":peach:", layout='wide')
with st.sidebar:
    st.markdown("## Welcome to Quality Control Tool for Georgia Continuous Count Station (CCS) data")
    st.markdown("#### This application is developed by [Smart Mobility and Infrastructure Lab](http://smil.engr.uga.edu/) at UGA")
    st.markdown("---")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 300px;
        margin-left: -300px;
    }
     
    """,
    unsafe_allow_html=True,
)

col1, col2, col3,col4,col5 = st.columns(5)

with col1:
    st.write(' ')
with col2:
    st.write(' ')
with col3:
    st.write(' ')
with col4:
    st.image("icon.png", width=500)
with col5:
    st.write(' ')

st.subheader('Clustering for CCS in Georgia :peach:')
#st.subheader('All clusters visualized on Map')

#导入母地图模块
DF = pd.read_csv('./clustering_result.csv')
geo = './map.geojson'
file = open(geo,encoding='utf8')
text = file.read()

markercluster_list = map_generator(geo,DF)


#load the sequence to be detected 
## ID, date, cluster, map
st.markdown('---')
st.subheader('kriging-based quality control for CCS sequence :white_check_mark:')

file=st.file_uploader('Please upload a csv file', type={'csv'})
if file is not None:
    file_df = pd.read_csv(file)

    stationname = file_df.station[0]
    Date = file_df.Date[0]
    cluster_belong = DF.loc[DF['stationname']==file_df.station[0]]['cluster'].values[0]
    other_stations_ls = DF.loc[DF['cluster'] ==cluster_belong]['stationname'].values.tolist()
    other_stations = ','.join(other_stations_ls)



    with st.expander("See Input Sequence Information"):
        
        tab1, tab2, tab3,tab4,tab5 = st.tabs([":blue[Station name]",":blue[Date]", ":blue[Cluster]", ":blue[Correlated stations]", ":blue[Cluster location]"])

        with tab1:
            st.markdown(stationname)
        
        with tab2:
            
            format_data = "%m/%d/%y"
            Date= datetime.datetime.strptime(Date, format_data)
            Date= Date.isoformat()[0:10]
            st.markdown(Date)
        
        with tab3:
            st.markdown(cluster_belong)
        
        with tab4:
            st.markdown(other_stations)
        
        with tab5:
            cluster_map= folium.Map(location=[DF.lat.mean(), DF.lon.mean()],zoom_start=6, control_scale=True)
            GeoJson(text).add_to(cluster_map)

            markercluster_list[cluster_belong].add_to(cluster_map)
            folium.LayerControl().add_to(cluster_map)
            st_data = folium_static(cluster_map,width=500,height=400)

        css = '''
        <style>
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size:1.0rem;
            }

        div[data-testid="stExpander"] div[role="button"] p {
            font-size: 1.0rem;
            }

        </style>
        '''
        st.markdown(css, unsafe_allow_html=True)


else:
    st.warning('you need to upload a sequence csv file.')


st.markdown(
    """
<style>
[data-testid="stTabs"] {
    font-size: 20px;
    color: Red;
}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown("""
<style>
div[data-testid="metric-container"] {
   background-color: rgba(28, 131, 225, 0.1);
   border: 2px solid rgba(28, 131, 225, 0.1);
   padding: 5% 5% 5% 15%;
   border-radius: 20px;
   color: rgb(30, 103, 119);
   overflow-wrap: break-word;
}

/* breakline for metric text         */
div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
   overflow-wrap: break-word;
   white-space: break-spaces;
   font-size: 20px;
   color: Red;
}
</style>
"""
, unsafe_allow_html=True)

###多选站点模块


if file is not None:
    dir_path = './database'
    path = dir_path+'/'+ Date[0:7]+'.csv'
    
    if os.path.exists(path):
      
        sequences = show_plot(path=path, station_options=other_stations_ls,date_option=Date,stationname=stationname)
        z = 16 #hidden dimension for graph convolution
        K = 1 #If using diffusion convolution, the actual diffusion convolution step is K+1 ##try more Convolution 
        h = 288
        model_path = './model/DGCN_811.pth'
        DGCN = DGCN(h, z, K)
        DGCN.load_state_dict(torch.load(model_path))
        model_processing(z,K,h,model_path,DGCN,sequences,cluster_belong,DF,stationname,other_stations_ls,file_df)
       
    
    else:
        uploaded_files = st.file_uploader("Please upload all correlated sequences", accept_multiple_files=True)
        if uploaded_files is not None:
            sequences=[]
            for uploaded_file in uploaded_files:
                add_file_df = pd.read_csv(uploaded_file)
                sequences.append(add_file_df)
            if len(sequences)==0:
                st.warning('Please upload correct files')
            else:
                sequences = show_plot_1(sequences,stationname)
                z = 16 #hidden dimension for graph convolution
                K = 1 #If using diffusion convolution, the actual diffusion convolution step is K+1 ##try more Convolution 
                h = 288
                model_path = './model/DGCN_811.pth'
                DGCN = DGCN(h, z, K)
                DGCN.load_state_dict(torch.load(model_path))
                model_processing(z,K,h,model_path,DGCN,sequences,cluster_belong,DF,stationname,other_stations_ls,file_df)
    
        else:
            st.warning('Please upload files of all CCS sequences in the cluster!')


else:
    st.warning('wait...')





# if file is not None:
#     z = 16 #hidden dimension for graph convolution
#     K = 1 #If using diffusion convolution, the actual diffusion convolution step is K+1 ##try more Convolution 
#     h = 288
#     model_path = '/Users/yongcanhuang/streamlit/clustering&kriging/model/DGCN_811.pth'
#     DGCN = DGCN(h, z, K)
#     DGCN.load_state_dict(torch.load(model_path))
#     X_res=model_processing(z,K,h,model_path,DGCN,sequences,cluster_belong,DF,stationname,other_stations_ls)
#     #X_res[0]-sequences[position]['Volume'].values
#     file_df.iloc[0][4:4+288]=X_res[0]

#     def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#       return df.to_csv().encode('utf-8')

#     csv = convert_df(file_df)

#     st.download_button(
#         label="Download data as CSV",
#         data=csv,
#         file_name='revised_sequence.csv',
#         mime='text/csv',
#     )
  

# else:
#     pass


