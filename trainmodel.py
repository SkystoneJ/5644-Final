from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

X_train = pd.read_csv('trainX.csv')
y_train = pd.read_csv('trainy.csv')
X_test = pd.read_csv('testX.csv')
y_test = pd.read_csv('testy.csv')
# 初始化随机森林回归器
regressor = RandomForestRegressor(n_estimators=400, random_state=42, max_depth=20, min_samples_split=3,
                                  min_samples_leaf=1, n_jobs=-1)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {rmse}')



