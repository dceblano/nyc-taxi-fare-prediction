import pandas as pd
from sklearn.impute import SimpleImputer
from read_parquet import get_parquet

# Load the dataset
df = get_parquet()

def perform_cleanup(df):
    # Maintaning data consistency for string data type
    # Convert all object and category type columns to strings
    for column in df.select_dtypes(include=['object', 'category']).columns:
        df[column] = df[column].astype(str)

    # Combining columns that has similar values (Airport Fee)
    # handle missing values before combining columns for Airport Fee
    if 'Airport_fee' in df.columns and 'airport_fee' in df.columns:
        # Fill missing values in each column before combining, will get an error with the imputation
        df['Airport_fee'] = df['Airport_fee'].fillna(0)
        df['airport_fee'] = df['airport_fee'].fillna(0)
        df['Total_Airport_Fee'] = df['Airport_fee'] + df['airport_fee']
        df.drop(['Airport_fee', 'airport_fee'], axis=1, inplace=True)

    # Separate imputation for numerical and categorical data
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    num_imputer = SimpleImputer(strategy='mean')
    df[numerical_columns] = num_imputer.fit_transform(df[numerical_columns])

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

    # Checking if the imputation is successful
    print("\nNull values in each column:")
    print(df.isnull().sum())

    # Remove duplicates
    duplicated_rows = df[df.duplicated(keep=False)]
    print("\nDuplicated rows:")
    print(duplicated_rows)
    df = df.drop_duplicates()
    print("\nRemoved duplicate rows.")

    # Data description
    print("Data Description:")
    print(df.describe(include='all'))  # describe all columns

    return df

# Checking the DF
clean_df = perform_cleanup(df)
print(clean_df)