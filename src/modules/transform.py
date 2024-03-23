import pandas as pd
import time

def data_transform(df):
    
    # Record the start time for runtime calculation.
    start_time = time.time()  # Record the start time

    df = df.copy()
    # Working with a copy of the DataFrame
    target = 'earnings'

    features_to_drop = ['VendorID', 'store_and_fwd_flag', 'extra', 'mta_tax', 'tolls_amount', 'improvement_surcharge', 'total_amount', 'Total_Airport_Fee']

    # Calculate Target Feature: Add fare amount and tip amount
    df[target] = df['fare_amount'] + df['tip_amount']

    # Drop Irrelevant features
    df.drop(features_to_drop, axis=1, inplace=True)

    print(df.info())

    # Get statistical details
    data_description = df[target].describe()

    # Print or display the statistical information
    print('\nStatistical Information:') 
    print(data_description)

    # Record the end time for runtime calculation and print the runtime.
    end_time = time.time()  # Record the end time
    runtime = end_time - start_time  # Calculate the elapsed time
    print("Runtime[data_transform]:", runtime, "seconds\n")  # Print the runtime

    return df   