# Cluster-informed CCS Data Quality Control

This Streamlit web application is developed by the Smart Mobility and Infrastructure Lab at UGA for Quality Control of Georgia Continuous Count Station (CCS) data.

## Features

- Clustering visualization for CCS in Georgia
- Kriging-based quality control for CCS sequences
- Interactive map visualization
- Data upload and processing capabilities

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Upload test1.csv and the 011-0103_S.csv from data folder to test the kriging results. 
   
   

3. Run the application:
```bash
streamlit run main.py
```

## Usage

1. Upload a CSV file containing CCS data
2. View the station information including:
   - Station name
   - Date
   - Cluster
   - Correlated stations
   - Cluster location on map
3. You have to upload the prompted all station CSV files in a group and it will return the repaired sequence CSV

## Data Requirements

The input CSV file should contain the following columns:
- station
- Date
- Volume data

## Development

This application is developed by [Smart Mobility and Infrastructure Lab](http://smil.engr.uga.edu/) at UGA. 
