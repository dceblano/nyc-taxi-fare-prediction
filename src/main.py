import pandas as pd
import datetime as dt

from modules.read_parquet import get_parquet
from modules.preprocessing import perform_cleanup
from modules.preprocessing import handle_outliers
from modules.transform import data_transform


def main():
    # Get parquet
    print("#1 Read parquet")
    taxi_df = get_parquet()
    print(taxi_df.info())

    # Perform cleanup
    print("#2 Perform cleanup")
    cleaned_df = perform_cleanup(taxi_df)

    # Handle Outlier
    print("#3 Handle Outlier")
    df_no_outliers = handle_outliers(taxi_df)

    # Engineer New Features & Transformations
    print("#4 Engineer New Features & Transformations")
    transformed_df = data_transform(df_no_outliers)   


if __name__ == "__main__":
    main()
