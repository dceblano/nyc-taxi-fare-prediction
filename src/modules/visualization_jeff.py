
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from read_parquet import get_parquet
from Preprocessing2 import perform_cleanup

# Load the dataset
print("Reading Parquet Files")

# Calling functions for data loading and clean up
df = get_parquet()
df = perform_cleanup(df)

# Convert datetime columns to pandas datetime objects
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

# Calculate trip duration in minutes
df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

# Created numerical_columns variable to contain features that has numerical values 
numerical_columns = ['passenger_count', 'trip_distance', 'fare_amount', 'extra', 'improvement_surcharge', 'tip_amount', 'tolls_amount', 'total_amount', 'congestion_surcharge', 'trip_duration','Total_Airport_Fee']

### Features Histogram ###

# Setting the overall figure size
plt.figure(figsize=(15, 15))
# Plot histograms for each numerical column using a for loop
for index, column in enumerate(numerical_columns):
    plt.subplot(4, 3, index + 1)
    plt.hist(df[column], bins=10)
    plt.grid(True)
    plt.title(column)
    plt.ylabel('Frequency')
    plt.xlabel('Value')
# Using tight_layout with a padding specification
plt.tight_layout(pad=3.0)
# Spacing Adjustment for Visuals
plt.subplots_adjust(wspace=0.5, hspace=0.5) 

### Correlation Heatmap ###

# Created a new variable for correlation computation
filtered_df = df[numerical_columns]
# Calculate the correlation matrix
corr = filtered_df.corr()
# Set up the matplotlib figure
plt.figure(figsize=(12, 10))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
# Add title
plt.title('Correlation Heatmap')
# Show the plot
plt.show()



# print(df['passenger_count'].value_counts().sort_index())

# # Plot with appropriate bins and a log scale.
# plt.figure(figsize=(10, 6))
# plt.hist(df['passenger_count'])
# plt.title('Passenger Count Distribution')
# plt.xlabel('Passenger Count')
# plt.ylabel('Log Frequency')
# plt.grid(True)
# plt.show()



