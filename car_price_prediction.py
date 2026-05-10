# =========================================================
# CodeAlpha Internship Project
# Advanced Car Price Prediction System
# Developed by : Shaik Numaan
# =========================================================
# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
# =========================================================
# Load Dataset
# =========================================================
cars = pd.read_csv("car data.csv")
print("\n========== DATASET PREVIEW ==========\n")
print(cars.head())
print("\n========== DATASET INFORMATION ==========\n")
print(cars.info())
print("\n========== MISSING VALUES ==========\n")
print(cars.isnull().sum())
# =========================================================
# Feature Engineering
# =========================================================
# Create Car Age Feature
cars["Car_Age"] = 2025 - cars["Year"]
# Remove unnecessary columns
cars.drop(["Year", "Car_Name"], axis=1, inplace=True)
# =========================================================
# Convert Categorical Data
# =========================================================
encoder = LabelEncoder()
cars["Fuel_Type"] = encoder.fit_transform(cars["Fuel_Type"])
cars["Selling_type"] = encoder.fit_transform(cars["Selling_type"])
cars["Transmission"] = encoder.fit_transform(cars["Transmission"])
# =========================================================
# Features and Target
# =========================================================
X = cars.drop("Selling_Price", axis=1)
y = cars["Selling_Price"]
# =========================================================
# Split Dataset
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("\nTraining Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)
# =========================================================
# Linear Regression Model
# =========================================================
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)
# =========================================================
# Random Forest Model
# =========================================================
forest_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)
forest_model.fit(X_train, y_train)
forest_predictions = forest_model.predict(X_test)
# =========================================================
# Model Evaluation
# =========================================================
lr_mae = mean_absolute_error(y_test, linear_predictions)
lr_r2 = r2_score(y_test, linear_predictions)
rf_mae = mean_absolute_error(y_test, forest_predictions)
rf_r2 = r2_score(y_test, forest_predictions)
print("\n========== LINEAR REGRESSION ==========")
print("Mean Absolute Error :", lr_mae)
print("R2 Score :", lr_r2)
print("\n========== RANDOM FOREST ==========")
print("Mean Absolute Error :", rf_mae)
print("R2 Score :", rf_r2)
# =========================================================
# Best Model
# =========================================================
if rf_r2 > lr_r2:
    print("\nBest Model : Random Forest Regressor")
else:
    print("\nBest Model : Linear Regression")
# =========================================================
# DIAGRAM 1
# Actual vs Predicted Graph
# =========================================================
plt.figure(figsize=(8,6))
plt.scatter(
    y_test,
    forest_predictions
)
plt.xlabel("Actual Car Prices")
plt.ylabel("Predicted Car Prices")
plt.title("Actual vs Predicted Car Price")
plt.grid(True)
plt.show()
# =========================================================
# DIAGRAM 2
# Feature Importance Graph
# =========================================================
importance = forest_model.feature_importances_
feature_names = X.columns
sorted_index = np.argsort(importance)
plt.figure(figsize=(10,6))
plt.barh(
    feature_names[sorted_index],
    importance[sorted_index]
)
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Car Price Prediction")
plt.grid(True)
plt.show()
# =========================================================
# Final Message
# =========================================================
print("\nProject Executed Successfully")
print("Car Price Prediction Completed")