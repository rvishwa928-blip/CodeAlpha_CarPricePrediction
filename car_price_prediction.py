# =========================================
# Car Price Prediction using Machine Learning
# (Auto CSV Creation)
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------
# 0. Auto-create CSV if not exists
# -----------------------------------------
csv_file = "car_data.csv"

if not os.path.exists(csv_file):
    print("CSV file not found. Creating car_data.csv automatically...")

    data = {
        "Brand": [
            "Maruti", "Hyundai", "Honda", "Toyota", "Tata",
            "Maruti", "Hyundai", "Honda", "Toyota", "Tata",
            "Maruti", "Hyundai", "Honda", "Toyota", "Tata",
            "Maruti", "Hyundai", "Honda", "Toyota", "Tata"
        ],
        "Mileage": [
            22000, 18000, 20000, 15000, 17000,
            24000, 16000, 19000, 14000, 16500,
            21000, 17500, 18500, 15500, 16800,
            23000, 16500, 19500, 14500, 17200
        ],
        "Horsepower": [
            67, 82, 90, 98, 86,
            70, 85, 92, 100, 88,
            69, 84, 91, 99, 87,
            72, 86, 93, 101, 89
        ],
        "EngineSize": [
            1.0, 1.2, 1.5, 1.8, 1.2,
            1.0, 1.2, 1.5, 1.8, 1.2,
            1.0, 1.2, 1.5, 1.8, 1.2,
            1.0, 1.2, 1.5, 1.8, 1.2
        ],
        "Year": [
            2018, 2019, 2020, 2017, 2019,
            2016, 2018, 2019, 2016, 2018,
            2017, 2019, 2020, 2018, 2019,
            2016, 2018, 2019, 2016, 2018
        ],
        "Price": [
            350000, 450000, 550000, 650000, 500000,
            300000, 420000, 530000, 620000, 480000,
            340000, 460000, 560000, 660000, 510000,
            310000, 430000, 540000, 630000, 490000
        ]
    }

    df_auto = pd.DataFrame(data)
    df_auto.to_csv(csv_file, index=False)

    print("car_data.csv created successfully!\n")

# -----------------------------------------
# 1. Load Dataset
# -----------------------------------------
df = pd.read_csv(csv_file)
print(df.head())

# -----------------------------------------
# 2. Data Cleaning
# -----------------------------------------
df.dropna(inplace=True)

# Encode categorical feature
le = LabelEncoder()
df['Brand'] = le.fit_transform(df['Brand'])

# Feature Engineering
df['Car_Age'] = 2025 - df['Year']
df.drop('Year', axis=1, inplace=True)

# -----------------------------------------
# 3. Feature Selection
# -----------------------------------------
X = df.drop('Price', axis=1)
y = df['Price']

# -----------------------------------------
# 4. Train-Test Split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------
# 5. Model Training
# -----------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------------------
# 6. Prediction
# -----------------------------------------
y_pred = model.predict(X_test)

# -----------------------------------------
# 7. Model Evaluation
# -----------------------------------------
print("\nMAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -----------------------------------------
# 8. Visualization
# -----------------------------------------
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()

# -----------------------------------------
# 9. Sample Prediction
# -----------------------------------------
sample_car = [[2, 15000, 120, 1.5, 3]]
predicted_price = model.predict(sample_car)
print("\nPredicted Car Price:", predicted_price[0])
