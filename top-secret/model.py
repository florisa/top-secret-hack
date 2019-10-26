import numpy as np
import pandas as pd

assembly_data = pd.read_excel("Daten.xlsx", sheet_name = "dataset1")
initial_data = pd.read_excel("Daten.xlsx", sheet_name = "initialinspection")
final_data = pd.read_excel("Daten.xlsx", sheet_name = "finalinspection")
result_data = pd.read_excel("Daten.xlsx", sheet_name = "Final")

# extracting specific columns
aux_1 = initial_data.iloc[:,-3:]
targets = initial_data.iloc[:,[0,2,8,9,10,11,12]]

# inputs and outputs
features = pd.concat([assembly_data.iloc[:,2:], aux_1], axis = 1)

# One Hot Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
targets.iloc[:,-1] = label_encoder.fit_transform(targets.iloc[:,-1])
one_hotencoder = OneHotEncoder(categorical_features = [-1])
targets = pd.DataFrame(one_hotencoder.fit_transform(targets).toarray())

# Missing Data
features = features.fillna(method = "ffill")
targets = targets.fillna(method = "ffill")

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Model Construction
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size = 0.25, random_state = 42)
model = RandomForestRegressor(n_estimators = 10, random_state = 42)
model.fit(x_train, y_train)