import pandas as pd
from sklearn.impute import SimpleImputer
import time

def perform_cleanup(df):
    start_time = time.time()  # Record the start time

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

    # Check number of records before removing duplicates
    num_records_before = len(df)

    # Remove duplicates
    duplicated_rows = df[df.duplicated(keep=False)]
    print("\nDuplicated rows:")
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

def data_scope(df):
    """
    Filters the DataFrame to include only payment types with values 1 and 2 and positive fare_amount.

    Parameters:
        df (DataFrame): The original DataFrame containing taxi fare data.

    Returns:
        DataFrame: Filtered DataFrame containing payment types 1 and 2 with positive fare_amount.

    Features Used:
        - fare_amount: Column representing the fare amount.
        - payment_type: Column representing the payment type.

    """
    fare = 'fare_amount'
    paytype = 'payment_type'

    # Filter DataFrame to include only payment types with values 1 and 2 and positive fare_amount
    scope_df = df[(df[paytype].isin([1, 2])) & (df[fare] > 0)]

    return scope_df



def handle_outliers(df):
    '''Handle outliers in a DataFrame column using the IQR method.
    
    Args:
    - df: DataFrame containing the data.

    Returns:
    - df_no_outliers: DataFrame with outliers removed.'''

    # Record the start time for runtime calculation.
    start_time = time.time()

    # Creating variable for the feature
    fare = 'fare_amount'

    # Get the data scope by filtering payment types 1 and 2 with positive fare amounts.
    scope_df = data_scope(df)

    # Count number of entries
    count = len(df)

    # Final count after filtering payment types and negative fare amounts
    final_count = len(scope_df)

    # Check how many entries where removed
    removed = count - final_count 

    # Print the count of entries before and after filtering.
    print('\nNumber entries', count)
    print('Payment Type 1 or 2 & Fare Amount greater than 0 :', final_count)
    print('Total entries removed', removed)

    # Print statistical details of fare_amount after filtering.
    data_desc = scope_df[fare].describe()
    print('\nStatistical Details:')
    print(data_desc)

    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = scope_df[fare].quantile(0.25)
    Q3 = scope_df[fare].quantile(0.75)

    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find maximum and minimum value
    max_value = scope_df[fare].max()
    min_value = scope_df[fare].min()

    # Print outlier boundaries and maximum/minimum values of fare_amount.
    print('\nLower Bound:', lower_bound)
    print('Upper Bound:', upper_bound)
    print("\nMaximum value in column '{}' is: {}".format(fare, max_value))
    print("Minimum value in column '{}' is: {}".format(fare, min_value))

    # Filter the DataFrame to remove outliers
    df_no_outliers = scope_df[(scope_df[fare] >= lower_bound) & (scope_df[fare] <= upper_bound)]

    # Print the number of records before and after removing outliers, and the number of outliers.
    print("\nNumber of records before removing outliers:", len(scope_df))
    print("Number of records after removing outliers:", len(df_no_outliers))
    print("Number of outliers", len(scope_df)-len(df_no_outliers))

    # Find maximum and minimum value
    max_value = df_no_outliers[fare].max()
    min_value = df_no_outliers[fare].min()

    print("\nAfter Removing Outliers:")

    # Get statistical details
    data_description = df_no_outliers[fare].describe()

    # Print or display the statistical information
    print('\nStatistical Information:')
    print(data_description)

    # Print maximum and minimum values of fare_amount after removing outliers.
    print("\nMaximum value in column '{}' is: {}".format(fare, max_value))
    print("Minimum value in column '{}' is: {}".format(fare, min_value))
    
    # Record the end time for runtime calculation and print the runtime.
    end_time = time.time()  # Record the end time
    runtime = end_time - start_time  # Calculate the elapsed time
    print("\nRuntime[handle_outliers]:", runtime, "seconds\n")  # Print the runtime

    return df_no_outliers


def extract_csv(df):
    # Created this method to extract sample dataset for data exploration
    # Get top 50 data from DataFrame
    top_50_data = df.head(50)
    # Specify the path where you want to save the CSV file
    csv_file_path = r"C:\Users\Administrator\OneDrive - Lambton College\Desktop\AI - Project\outliers.csv"

    # Save the top 50 data to a CSV file
    top_50_data.to_csv(csv_file_path, index=False)

    # print("Top 50 data extracted to CSV file:", csv_file_path)
    print("Top 50 data extracted to CSV file:", csv_file_path)
