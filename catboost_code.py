import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
data = fetch_california_housing()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = CatBoostRegressor(
    iterations=100, 
    learning_rate=0.1, 
    depth=6, 
    loss_function='RMSE',
    verbose=0, 
    random_seed=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE trên tập kiểm tra: {rmse:.4f}")
