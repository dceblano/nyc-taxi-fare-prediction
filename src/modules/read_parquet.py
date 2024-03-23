import pandas as pd
import datetime as dt
import time
from pathlib import Path
import os

def get_parquet():
    # Get the current working directory
    cwd = Path(os.getcwd())
    print("Current Path:", cwd)
    
    # Define the path to the 'data' directory
    data_dir = cwd / 'data/parquet'

    # Check if the path exists
    if not data_dir.exists():
        print(f"The directory {data_dir} does not exist.")
        return pd.DataFrame()  # Return an empty DataFrame if the directory doesn't exist

    # 2023 trip data dataframe
    tripdata_2023_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in data_dir.glob('*.parquet')
    )

    return tripdata_2023_df
