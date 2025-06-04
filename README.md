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

2. Download required data files:
   - Create a `database` directory in the project root
   - Download the following files and place them in the appropriate locations:
     - `database/2019-01.csv`
     - `database/2019-08.csv`
     - `database/2019-10.csv`
     - `adj_mat.npy`
     - `stationname.npy`
   
   Please contact the Smart Mobility and Infrastructure Lab for access to these data files.

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