import pandas as pd
import datetime as dt

from modules.read_parquet import get_parquet
from modules.preprocessing import perform_cleanup

def main():
    # Get parquet
    print("#1 Read parquet")
    taxi_df = get_parquet()
    print(taxi_df.info())

    # Perform cleanup
    print("#2 Perform cleanup")
    cleaned_df = perform_cleanup(taxi_df)

if __name__ == "__main__":
    main()
