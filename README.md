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

For normal mode:
```bash
streamlit run main.py
```

For debug mode (recommended):
```bash
KMP_DUPLICATE_LIB_OK=TRUE streamlit run main.py --logger.level=debug
```

Note: The debug mode is recommended as it:
- Fixes OpenMP runtime conflicts (common on macOS)
- Provides detailed logging for troubleshooting
- Shows additional information about data processing

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
- Volume data (288 time points, one for every 5 minutes in a day)

## Troubleshooting

If you encounter the following error:
```
OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
```
Use the debug mode command provided above with the `KMP_DUPLICATE_LIB_OK=TRUE` environment variable.

## Development

This application is developed by [Smart Mobility and Infrastructure Lab](http://smil.engr.uga.edu/) at UGA. 

## Repository Structure

```
.
├── data/                  # Data directory
│   ├── adj_mat.npy       # Adjacency matrix for station relationships
│   ├── test1.csv         # Sample test data
│   └── 011-0103_S.csv    # Sample station data
├── model_save/           # Saved model weights
├── main.py              # Main application file
├── model.py             # Model definitions
├── output.py            # Output processing
└── requirements.txt     # Python dependencies
```
