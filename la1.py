# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np


# %%
# Set seed for reproducibility
import random
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# %%
data=pd.read_csv("boston_housing.csv")
data.head()

# %%
data.isnull().sum()

# %%
data.info()

# %%
data.describe()

# %%
X = data.drop(columns=["MEDV"])  # Assuming 'MEDV' is the target column
y = data["MEDV"].values

# %%
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Normalize the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
# Build the Deep Neural Network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(13,), name='input-layer'),  # Input layer
    
    # First hidden layer with 128 units and ReLU activation
    tf.keras.layers.Dense(128, activation='relu', name='hidden-layer-1'),
    
    # Batch Normalization to stabilize training
    tf.keras.layers.BatchNormalization(name='batch-normalization-1'),
    
    # Dropout layer to prevent overfitting
    tf.keras.layers.Dropout(0.2, name='dropout-layer-1'),
    
    # Second hidden layer with 64 units and ReLU activation
    tf.keras.layers.Dense(64, activation='relu', name='hidden-layer-2'),
    
    # Batch Normalization for the second hidden layer
    tf.keras.layers.BatchNormalization(name='batch-normalization-2'),
    
    # Dropout layer to prevent overfitting
    tf.keras.layers.Dropout(0.2, name='dropout-layer-2'),
    
    # Output layer with 1 unit (no activation)
    tf.keras.layers.Dense(1, name='output-layer')
])

# %%
# Compile the model with Adam optimizer and MSE loss function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# %%
# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)


# %%
# Predict the prices using the trained model
y_pred = model.predict(X_test)

# %%
model.summary()

# %%
# Performance Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# %%
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R² Score: {r2}')
print(f'Mean Absolute Error (MAE): {mae}')

# %%


# %%
import seaborn as sns
sns.regplot(x=y_test, y=y_pred)
plt.title("Regression Line for Predicted values")
plt.show()

# %%
#Comparison with traditional approaches
#First let's try with a simple algorithm, the Linear Regression:
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# %%
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

# %%
print('Mean squared error on test data: ', mse_lr)
print('Mean absolute error on test data: ', mae_lr)
r2 = r2_score(y_test, y_pred_lr)
print('R2: ',r2)
rmse = np.sqrt(mse_lr) 
print('RMSE: ',rmse)


# %%
import numpy as np

# Example new input data (e.g., 5 houses with 13 features each)
new_data = np.array([
    [0.1, 0.2, 7.2, 0.0, 0.0, 6.0, 15.0, 1.0, 5.0, 300.0, 10.0, 300.0, 15.0],
    [0.3, 0.4, 8.0, 0.1, 0.1, 5.5, 16.0, 2.0, 4.5, 320.0, 9.0, 310.0, 16.0],
    [0.05, 0.1, 6.5, 0.1, 0.0, 7.0, 14.0, 1.5, 5.2, 310.0, 11.0, 280.0, 14.0],
    [0.2, 0.3, 7.8, 0.0, 0.1, 6.5, 17.0, 1.8, 4.8, 290.0, 10.0, 305.0, 15.5],
    [0.1, 0.2, 7.0, 0.0, 0.2, 6.8, 16.5, 1.2, 5.0, 310.0, 10.5, 300.0, 15.0]
])

# Ensure the data is in the correct format and shape
# Make predictions
predictions = model.predict(new_data)

# Output the predictions
print("Predicted housing prices:", predictions)


# %%
# Plotting training and validation loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
# Plotting predicted vs actual values
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Ideal line
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()


# %%
#checking the distribution of the target variable
import seaborn as sns
sns.distplot(data.MEDV)


# %%


# %%


# %%


# %%


# %%


# %%
