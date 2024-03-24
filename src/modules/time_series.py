import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from itertools import product
import time

def perform_timeseries_forecasting(df):    
    # Record the start time for runtime calculation.
    start_time = time.time()  # Record the start time

    df['day_of_week'] = df['tpep_pickup_datetime'].dt.day_name()
    df['hour_of_day'] = df['tpep_pickup_datetime'].dt.hour

    # Group by day of the week and hour of the day, and count the number of trips
    trips_per_hour_df = df.groupby(['day_of_week', 'hour_of_day']).size().reset_index(name='num_trips')

    # Create a complete datetime column based on the reference date
    day_mapping = { 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7 }
    trips_per_hour_df['day_of_week_numeric'] = trips_per_hour_df['day_of_week'].map(day_mapping)
    reference_date = pd.Timestamp('2023-01-01')
    trips_per_hour_df['datetime'] = reference_date + pd.to_timedelta(trips_per_hour_df['day_of_week_numeric'], unit='D') + pd.to_timedelta(trips_per_hour_df['hour_of_day'], unit='H')
   
    # Sort the date accordingly then delete unused features
    trips_per_hour_df.sort_values('datetime', inplace=True)
 
    #print('trips_per_hour_df')
    #print(trips_per_hour_df)
    print('Saturday 11AM:', trips_per_hour_df[(trips_per_hour_df['day_of_week'] == 'Saturday') & (trips_per_hour_df['hour_of_day'] == 11)])

    # Perform one-hot encoding
    one_hot_encoded_days = pd.get_dummies(trips_per_hour_df['day_of_week'], prefix='day', drop_first=False)
    one_hot_encoded_hours = pd.get_dummies(trips_per_hour_df['hour_of_day'], prefix='hour', drop_first=False)

    # Concatenate the one-hot encoded days dataframe with the original DataFrame
    trips_per_hour_df.reset_index(drop=True, inplace=True)
    one_hot_encoded_days.reset_index(drop=True, inplace=True)
    one_hot_encoded_days = one_hot_encoded_days.astype(int)
    trips_per_hour_df = pd.concat([trips_per_hour_df, one_hot_encoded_days], axis=1)

    # Concatenate the one-hot encoded hours dataframe with the original DataFrame
    trips_per_hour_df.reset_index(drop=True, inplace=True)
    one_hot_encoded_hours.reset_index(drop=True, inplace=True)
    one_hot_encoded_hours = one_hot_encoded_hours.astype(int)
    trips_per_hour_df = pd.concat([trips_per_hour_df, one_hot_encoded_hours], axis=1)
    
    # Extract features and target variable
    X = trips_per_hour_df[['hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 
            'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_18', 'hour_20', 'hour_21', 'hour_22', 'hour_23',   
            'day_Monday', 'day_Tuesday', 'day_Wednesday', 'day_Thursday', 'day_Friday', 'day_Saturday', 'day_Sunday']]
    y = trips_per_hour_df['num_trips']

    # Define the parameter grid for SARIMA model
    p_values = range(0, 3)  # Range of AR terms
    d_values = range(0, 2)  # Range of differences
    q_values = range(0, 3)  # Range of MA terms
    P_values = range(0, 3)  # Range of seasonal AR terms
    D_values = range(0, 2)  # Range of seasonal differences
    Q_values = range(0, 3)  # Range of seasonal MA terms
    s_values = [24]  # Seasonal period (assuming hourly data)
    # Generate all possible combinations of parameters
    param_grid = product(p_values, d_values, q_values, P_values, D_values, Q_values, s_values)

    # Initialize variables to store best model and its performance
    best_model = None
    best_params = None
    best_mse = float('inf')  # Initialize with a large value
    best_rmse = float('inf')  # Initialize with a large value

    # Initialize time series cross-validator
    tscv = TimeSeriesSplit(n_splits=2)

    # Perform grid search with cross-validation
    for params in param_grid:
        mse_scores = []

        # Perform cross-validation
        for train_index, test_index in tscv.split(X):
            # Split the data into training and testing sets
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            print("Data type of y_train:", type(y_train))
            print("Data type of X_train_encoded:", type(X_train))

            print("Shape of y_train:", y_train.shape)
            print("Shape of X_train_encoded:", X_train.shape)

            # Define SARIMA parameters
            #order = (1, 1, 1)  # ARIMA order
            #seasonal_order = (1, 1, 1, 24)  # Seasonal order (24 hours in a day for daily seasonality)

            try:
                sarima_model = SARIMAX(y_train, exog=X_train, order=params[:3], seasonal_order=params[3:])
                #sarima_model = SARIMAX(y_train, exog=X_train, order=order, seasonal_order=seasonal_order)
                sarima_result = sarima_model.fit()
            except np.linalg.LinAlgError as e:
                print(f"Error encountered for order={params[:3]}, seasonal_order={params[3:]}: {e}")
                break

            # Make predictions
            predictions = sarima_result.predict(start=len(X_test), end=len(X_test)+len(X_test)-1, exog=X_test, dynamic=False)
            
            # Calculate MSE
            mse = mean_squared_error(y_test, predictions)
            print('mse:', mse)
            # Calculate root mean squared error (RMSE)
            rmse = np.sqrt(mse)
            print(rmse)
            mse_scores.append(mse)
    
        # Compute the average MSE across all folds
        average_mse = sum(mse_scores) / len(mse_scores)
        print("Average Mean Squared Error (Cross-Validation):", average_mse)
        print("Average Root Mean Squared Error (Cross-Validation):", np.sqrt(average_mse))

        # Update best model if current model has lower MSE
        #'''
        if average_mse < best_mse:
            best_mse = average_mse
            best_rmse = np.sqrt(best_mse)
            best_params = params
            best_model = SARIMAX(y, exog=X, order=best_params[:3], seasonal_order=best_params[3:])
            sarima_result = best_model.fit()  # Fit the best model using the entire dataset
        #'''

    # Print the best model parameters and MSE
    print("Best SARIMA Model:")
    print("Order:", best_params[:3])
    print("Seasonal Order:", best_params[3:])
    print("Best MSE:", best_mse) 
    print("Best RMSE:", best_rmse)

    end_time = time.time()  # Record the end time
    runtime = end_time - start_time  # Calculate the elapsed time
    print("Runtime[Tme Series]:", runtime, "seconds\n")  # Print the runtime

    return sarima_result

def run_example_forecast(day_of_week, hour_of_day, time_series_model):
   
    exog_df = generate_exog(day_of_week, hour_of_day)

    # Call the forecast function of the SARIMAX model
    forecast = time_series_model.forecast(steps=1, exog=exog_df)
    print(f"Forecasted value is: {forecast}")
    return forecast

# Function to generate the exogenous variables DataFrame dynamically
def generate_exog(day_of_week, hour_of_day):
    # Create empty DataFrame with all zeros
    exog_data = {
        'day_Monday': [0],
        'day_Tuesday': [0],
        'day_Wednesday': [0],
        'day_Thursday': [0],
        'day_Friday': [0],
        'day_Saturday': [0],
        'day_Sunday': [0],
        'hour_0': [0],
        'hour_1': [0],
        'hour_2': [0],
        'hour_3': [0],
        'hour_4': [0],
        'hour_5': [0],
        'hour_6': [0],
        'hour_7': [0],
        'hour_8': [0],
        'hour_9': [0],
        'hour_10': [0],
        'hour_11': [0],
        'hour_12': [0],
        'hour_13': [0],
        'hour_14': [0],
        'hour_15': [0],
        'hour_16': [0],
        'hour_17': [0],
        'hour_18': [0],
        'hour_19': [0],
        'hour_20': [0],
        'hour_21': [0],
        'hour_22': [0],
        'hour_23': [0],
    }
    
    # Set the corresponding day and hour to 1
    exog_data[f'day_{day_of_week}'] = [1]
    exog_data[f'hour_{hour_of_day}'] = [1]
    
    # Create DataFrame with exogenous variables
    exog_df = pd.DataFrame(exog_data)
    return exog_df