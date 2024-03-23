import seaborn as sns
import matplotlib.pyplot as plt
import time

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
