# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python libraries
2. Load the dataset
3. Train the model for Ridge, Lasso and ElasticNet
4. Fit and Evaluate the model
5. Visualize the model

## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: Poojasree B
RegisterNumber: 212223040148
*/
# Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/encoded_car_data.csv'
df = pd.read_csv(file_path)

# Select relevant features and target variable
X = df.drop(columns=['price'])  # All columns except 'price'
y = df['price']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models and pipelines
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5)  # l1_ratio controls L1 vs L2 mix
}

# Iterate over models and evaluate
for name, model in models.items():
    pipeline = Pipeline([
        ("polynomial_features", PolynomialFeatures(degree=2)),
        ("regressor", model)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    print(f"\n{name} Regression Results:")
    print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
    print("R-squared:", r2_score(y_test, y_pred))

```

## Output:
![image](https://github.com/user-attachments/assets/42d0e30c-5e7a-4b80-95ce-05332e18175a)


## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
