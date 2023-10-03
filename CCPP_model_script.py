
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv('CCPP_data.csv')

# Features and target variable
X = df.drop('PE', axis=1)
y = df['PE']

# Split the data into training and test sets (70% training, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Perform 10-fold cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error')
rmse_cv = -np.mean(cv_scores)

# Train the Random Forest model on the entire training set
rf_model.fit(X_train, y_train)

# Predict the target variable on the test set
y_pred = rf_model.predict(X_test)

# Calculate the RMSE on the test set
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Cross-validation RMSE: {rmse_cv}')
print(f'Test RMSE: {rmse_test}')
