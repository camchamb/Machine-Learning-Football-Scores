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


# Base setup for the df

df = pd.read_csv('all_game_stats.csv')


# Configure/encode the data

pd.set_option("display.max_rows", None)

# Drop less vital features

columns_to_drop = ['kickingPoints_team1','kickingPoints_team2','schoolId_team1', 'schoolId_team2', 'defensiveTDs_team1', 'defensiveTDs_team2', 'interceptionTDs_team1', 'kickReturnTDs_team1', 'passingTDs_team1', 'puntReturnTDs_team1', 'rushingTDs_team1', 'interceptionTDs_team2', 'kickReturnTDs_team2', 'passingTDs_team2', 'puntReturnTDs_team2', 'rushingTDs_team2']

df = df.drop(columns=columns_to_drop)


#encoding the data. ALL NORMALIZATION AND ENCODING CODE DONE BY TEAMMATE. I help to develop as well.

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

string_cols = ['school_team1','conference_team1','school_team2','conference_team2','homeAway_team1','homeAway_team2',]

for col in string_cols:
    df[col] = encoder.fit_transform(df[[col]])


# Calculate the ratio Ex: instead off 22-40, it would be 22/40 so that it is usable

hyphen_cols = ['thirdDownEff_team1','fourthDownEff_team1','fourthDownEff_team2','thirdDownEff_team2','completionAttempts_team1','completionAttempts_team2','totalPenaltiesYards_team1', 'totalPenaltiesYards_team2']

for col in hyphen_cols:
    # Replace empty slots with 0-0.
    df[col] = df[col].replace('', '0-0')
    try:
        df[[f'{col}_num', f'{col}_den']] = df[col].str.split('-', expand=True).astype(float)
    except ValueError:
        df[[f'{col}_num', f'{col}_den']] = [[0.0, 1.0]] * len(df)
    df[f'{col}_ratio'] = df[f'{col}_num'] / df[f'{col}_den']
    df = df.drop(columns=[col, f'{col}_num', f'{col}_den'])


time_cols = ['possessionTime_team1','possessionTime_team2']

def time_to_seconds(time_str):
    time_str = str(time_str)
    parts = time_str.split(':')
    if (len(parts) < 2): return 0
    minutes, seconds = parts[0], parts[1];
    return int(minutes) * 60 + int(seconds)

for col in time_cols:
  df[col] = df[col].apply(time_to_seconds)

# Setting up test split of 20%

x, y = df.drop(columns=['points_team1', 'points_team2']), df[['points_team1', 'points_team2']]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# Normalizing the data

X_train = pd.DataFrame(X_train).replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = pd.DataFrame(X_test).replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)


# Impute Missing Data

imputer = SimpleImputer(strategy='mean')  # You can use 'median' or 'most_frequent' as well

imputer.fit(X_train)

X_train_index = X_train.index
X_train_columns = X_train.columns
X_test_index = X_test.index
X_test_columns = X_test.columns


# Transform both training and testing data

X_train = imputer.transform(X_train.values)
X_test = imputer.transform(X_test.values)


# Added because it kept converting it to an np_array

X_train = pd.DataFrame(X_train_scaled, index=X_train_index, columns=X_train_columns)
X_test = pd.DataFrame(X_test_scaled, index=X_test_index, columns=X_test_columns)

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
