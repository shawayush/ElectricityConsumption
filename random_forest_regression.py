# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset1 = pd.read_csv('bill.csv')
X1 = dataset1.iloc[:, 0:1].values
y1 = dataset1.iloc[:, 1].values


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X1, y1)

# Predicting a new result


# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X1), max(X1), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X1, y1, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Electricity Bill (Random Forest Regression)')
plt.xlabel('Time duration of 3 months')
plt.ylabel('price')
plt.show()