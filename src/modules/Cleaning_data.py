import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from read_parquet import get_parquet
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Load the dataset
print("Reading Parquet Files")
df = get_parquet()
print("Reading Successfully!")

def perform_cleanup(df):
    start_time = time.time()  # Record the start time

    # Maintaning data consistency for string data type
    # Convert all object and category type columns to strings
    for column in df.select_dtypes(include=['object', 'category']).columns:
        df[column] = df[column].astype(str)
    print("Converting Columns Successfully")

    # Combining similar columns (e.g., Airport Fee) and handling missing values
    print("Combinining Columns")
    if 'Airport_fee' in df.columns and 'airport_fee' in df.columns:
        df['Airport_fee'] = df['Airport_fee'].fillna(0)
        df['airport_fee'] = df['airport_fee'].fillna(0)
        df['Total_Airport_Fee'] = df['Airport_fee'] + df['airport_fee']
        df.drop(['Airport_fee', 'airport_fee'], axis=1, inplace=True)
    print("Successfully Combined Columns")

    print("Separating Numerical and Categorical Data")
    # Separating numerical and categorical data
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    print("Separation Complete")

    print("Starting Imputation for Numerical Columns")
    # Imputation for numerical columns
    knn_imputer = KNNImputer(n_neighbors=1)
    df[numerical_columns] = knn_imputer.fit_transform(df[numerical_columns])
    print("Completed Imputation for Numerical Columns")

    print("Starting Imputation for Categhorical Columns")
    # Encoding categorical columns, imputation, and decoding
    if len(categorical_columns) > 0:
        encoder = OrdinalEncoder()
        # Fit and transform the data to encode
        encoded_cats = encoder.fit_transform(df[categorical_columns].fillna('Missing'))
        # Impute the encoded data
        imputed_cats = knn_imputer.fit_transform(encoded_cats)
        # Inverse transform to decode back to original categories
        decoded_cats = encoder.inverse_transform(imputed_cats)
        df[categorical_columns] = decoded_cats
    print("Completed Imputation for Categorical Columns")

    # Check for null values post-imputation
    print("\nNull values in each column:")
    print(df.isnull().sum())

    # Check number of records before removing duplicates
    num_records_before = len(df)

    # Remove duplicates
    print("\nDuplicated rows:")
    duplicated_rows = df[df.duplicated(keep=False)]
    print(duplicated_rows)
    df = df.drop_duplicates()

    # Check number of records after removing duplicates
    num_records_after = len(df)

    print("\nRemoved duplicate rows.")
    print("\nNumber of records before removing duplicates:", num_records_before)
    print("Number of records after removing duplicates:", num_records_after)


    # Data description
    print("\nData Description:")
    print(df.describe(include='all'))  # describe all columns

    print(df.info())
    print(df.head())

    end_time = time.time()  # Record the end time
    runtime = end_time - start_time  # Calculate the elapsed time
    print("\nRuntime[perform_cleanup]:", runtime, "seconds\n")  # Print the runtime

    return df

# Checking the DF
clean_df = perform_cleanup(df)
print(clean_df)