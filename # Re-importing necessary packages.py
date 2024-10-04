# Re-importing necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Reloading the data from the uploaded Excel file
file_path = '/mnt/data/ecsa特征及目标.xlsx'
data = pd.read_excel(file_path)

# Renaming columns as done previously
data.rename(columns={
    'Temperature (°C)': 'Temperature',
    'Current Density (A/cm²)': 'Current_Density',
    'Loading (mg/cm²)': 'Loading',
    'ECSA (% of Initial)': 'ECSA_Initial'
}, inplace=True)

# Assuming the data is divided into four experiments
n = len(data) // 4  # Approximate number of rows per experiment
data['Experiment_ID'] = [1]*n + [2]*n + [3]*n + [4]*(len(data) - 3*n)

# Separate features and target variable
X = data[['Temperature', 'Current_Density', 'Loading', 'Cycles', 'Experiment_ID']]
y = data['ECSA_Initial']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (scale them to have mean 0 and variance 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train the MLP model
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', random_state=42, max_iter=500)
mlp.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = mlp.predict(X_test_scaled)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

mse, mae, rmse
