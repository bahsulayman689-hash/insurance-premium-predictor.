import pandas as pd
import numpy as np
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error , r2_score
import warnings as wr
wr.filterwarnings('ignore')

import random
random.seed(0)
np.random.seed(0)

insurance_data = pd.read_csv("insurance.csv")
print(insurance_data.head())
print(insurance_data.shape)
print(insurance_data.tail())
insurance_data.info()
print(insurance_data.isnull().sum())
print(insurance_data.duplicated().sum())
insurance_data = insurance_data.drop_duplicates()
print(insurance_data.duplicated().sum())
print(insurance_data["sex"].value_counts())

insurance_data[insurance_data["sex"] == 1]
insurance_data[insurance_data["sex"] == 0]
print(insurance_data["sex"].value_counts())
insurance_data["sex"] = insurance_data["sex"].map({"male": 1, "female":0})
print(insurance_data.head())
insurance_data["smoker"] = insurance_data["smoker"].map({"yes": 1, "no":0})
print(insurance_data.head())
print(insurance_data["charges"].describe())

X = insurance_data.drop(["region", "charges"], axis=1)
Y = insurance_data["charges"]
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    )
print(X.shape)
print(X_train.shape)
print(X_test.shape)
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=300,
                              max_depth=10,
                              min_samples_split=12,
                              max_leaf_nodes=5,
                              random_state=42,
                              )
model.fit(X_train_scaler, Y_train)
Y_training_prediction = model.predict(X_train_scaler)
the_mean_square_error = mean_squared_error(Y_train, Y_training_prediction)
r2_error = r2_score(Y_train, Y_training_prediction)
print(f"the mean square error is {the_mean_square_error}")
print(f"the r2_error of the model is {r2_error} ")

Y_testing_prediction = model.predict(X_test_scaler)
the_mean_square_error = mean_squared_error(Y_test, Y_testing_prediction)
r2_error = r2_score(Y_test, Y_testing_prediction)
print(f"the mean square error is {the_mean_square_error}")
print(f"the r2_error of the model is {r2_error} ")

joblib.dump(model, "insurance_model.pkl")
joblib.dump(scaler, "insurance_scaler.pkl")
model = joblib.load("insurance_model.pkl")
scaler = joblib.load("insurance_scaler.pkl")