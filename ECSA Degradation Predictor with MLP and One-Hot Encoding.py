# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import ace_tools as tools

# Load the data from Excel file
file_path = '/mnt/data/ecsa特征及目标.xlsx'
data = pd.read_excel(file_path)

# Rename the columns for easier handling
data.rename(columns={
    'Temperature (°C)': 'Temperature',
    'Current Density (A/cm²)': 'Current_Density',
    'Loading (mg/cm²)': 'Loading',
    'ECSA (% of Initial)': 'ECSA_Initial'
}, inplace=True)

# Assume the data is divided into four experiments, and add an Experiment_ID column
n = len(data) // 4  # Approximate number of rows per experiment
data['Experiment_ID'] = [1]*n + [2]*n + [3]*n + [4]*(len(data) - 3*n)

# One-hot encode the Experiment_ID column
encoder = OneHotEncoder(sparse=False)
experiment_id_encoded = encoder.fit_transform(data[['Experiment_ID']])

# Combine the one-hot encoded Experiment_ID with the other features
X = np.hstack((data[['Temperature', 'Current_Density', 'Loading', 'Cycles']].values, experiment_id_encoded))
y = data['ECSA_Initial'].values

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (excluding the one-hot encoded Experiment_ID columns)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[:, :4])  # Scale only the first 4 columns (Temperature, Current Density, Loading, Cycles)
X_train_scaled = np.hstack((X_train_scaled, X_train[:, 4:]))  # Combine scaled features with one-hot encoded Experiment_ID
X_test_scaled = scaler.transform(X_test[:, :4])
X_test_scaled = np.hstack((X_test_scaled, X_test[:, 4:]))

# Build and train the MLP model
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', random_state=42, max_iter=1000)
mlp.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = mlp.predict(X_test_scaled)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Prepare new data for prediction (88°C, 5A/cm², 0.3mg/cm²)
new_data = pd.DataFrame({
    'Temperature': [88] * len(data['Cycles'].unique()),
    'Current_Density': [5] * len(data['Cycles'].unique()),
    'Loading': [0.3] * len(data['Cycles'].unique()),
    'Cycles': data['Cycles'].unique()
})

# Encode new Experiment_ID for prediction (using an existing Experiment_ID = 4)
new_experiment_id_encoded = encoder.transform([[4]] * len(new_data))

# Standardize new data features
new_data_scaled = scaler.transform(new_data[['Temperature', 'Current_Density', 'Loading', 'Cycles']])
new_data_scaled = np.hstack((new_data_scaled, new_experiment_id_encoded))

# Predict ECSA for the new data
ecsa_predictions = mlp.predict(new_data_scaled)

# Combine the predicted ECSA values with corresponding cycle numbers for display
predicted_ecsa = pd.DataFrame({
    'Cycles': new_data['Cycles'],
    'Predicted ECSA': ecsa_predictions
})

# Display the predicted ECSA values and model evaluation metrics
tools.display_dataframe_to_user(name="Predicted ECSA for 88°C, 5A, 0.3mgIr (with One-Hot)", dataframe=predicted_ecsa)

# Print model evaluation metrics
print(f"Model Evaluation:\nMSE: {mse}\nMAE: {mae}\nRMSE: {rmse}")
