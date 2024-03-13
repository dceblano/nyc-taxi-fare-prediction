import pandas as pd
import datetime as dt
import time
from pathlib import Path
import os

def get_parquet():
    cwd = os.getcwd()
    print( "Current Path:", cwd )
    os.chdir('data')
    data_dir = Path(os.getcwd() +'/parquet/')

    # 2023 trip data dataframe
    tripdata_2023_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in data_dir.glob('*.parquet')
    )

    return tripdata_2023_df 