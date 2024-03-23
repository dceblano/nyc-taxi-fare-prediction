import seaborn as sns
import matplotlib.pyplot as plt
import time
import pandas as pd

def box_plot(scope_df, df_no_outliers):
    """
    Generates boxplots of fare_amount before and after handling outliers.

    Parameters:
        scope_df (DataFrame): DataFrame containing data before handling outliers.
        df_no_outliers (DataFrame): DataFrame containing data after handling outliers.

    Returns:
        None

    Plots:
        - Boxplot of 'fare_amount' before handling outliers.
        - Boxplot of 'fare_amount' after handling outliers.
        
    """
    start_time = time.time()  # Record the start time

    # Creating variables for the features 
    fare = 'fare_amount'

    # Plot boxplot of 'fare_amount' before handling outliers
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1) 
    sns.boxplot(y=scope_df[fare])
    plt.title('Boxplot of Fare Amount')
    plt.ylabel('Fare Amount ($)')

    # Plot boxplot of 'fare_amount' after handling outliers
    plt.subplot(1, 2, 2) 
    sns.boxplot(y=df_no_outliers[fare])
    plt.title('After Handling Outliers')
    plt.ylabel('Fare Amount ($)')

    # Show the plots
    plt.tight_layout()
    plt.show()
    
    end_time = time.time()  # Record the end time
    runtime = end_time - start_time  # Calculate the elapsed time
    print("\nRuntime[handle_outliers]:", runtime, "seconds\n")  # Print the runtime

    
def plot_features(df):
    # Convert datetime columns to pandas datetime objects
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

# Calculate trip duration in minutes
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

# Created numerical_columns variable to contain features that has numerical values 
    numerical_columns = [
        'passenger_count', 'trip_distance', 'fare_amount', 'extra',
        'improvement_surcharge', 'tip_amount', 'tolls_amount', 'total_amount',
        'congestion_surcharge', 'trip_duration', 'Total_Airport_Fee'
    ]

    # Plot histograms for each numerical column
    plt.figure(figsize=(15, 15))
    for index, column in enumerate(numerical_columns):
        plt.subplot(4, 3, index + 1)
        plt.hist(df[column], bins=10)
        plt.grid(True)
        plt.title(column)
        plt.ylabel('Frequency')
        plt.xlabel('Value')
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Calculate and plot correlation heatmap
    filtered_df = df[numerical_columns]
    corr = filtered_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
    plt.title('Correlation Heatmap')
    plt.show()