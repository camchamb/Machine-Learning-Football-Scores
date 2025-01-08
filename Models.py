import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression


### Best models ###

# MLP
mlp_best = MLPRegressor(hidden_layer_sizes=(100,), alpha=0.001, learning_rate='constant', learning_rate_init=0.01, max_iter=4000)

# Decision tree
dtr_best = DecisionTreeRegressor(min_samples_leaf = 25, min_samples_split = 6, max_depth = 9)

# KNN
knn_best = KNeighborsRegressor(n_neighbors=16, weights="distance")

# Random forest
rfr_best = RandomForestRegressor(n_estimators=200, min_samples_split=2, min_samples_leaf=1, max_depth=20)

# Gradient boosting
gbr_best = GradientBoostingRegressor()

# Ensemble

estimators = [('mlp', mlp_best),
              ('dtr', dtr_best),
              ('knn', knn_best),
              ('rfr', rfr_best),
              ('gbr', gbr_best)]

stack_ens = MultiOutputRegressor(StackingRegressor(estimators = estimators, final_estimator = LinearRegression(), n_jobs=-1), n_jobs=-1)

stack_ens.fit(X_train_scaled, y_train)

y_train_pred_stack_ens = stack_ens.predict(X_train_scaled)
y_test_pred_stack_ens = stack_ens.predict(X_test_scaled)

train_mae_stack_ens = mean_absolute_error(y_train, y_train_pred_stack_ens)
test_mae_stack_ens = mean_absolute_error(y_test, y_test_pred_stack_ens)


print('Train MAE: ', train_mae_stack_ens)
print('Test MAE: ', test_mae_stack_ens)
