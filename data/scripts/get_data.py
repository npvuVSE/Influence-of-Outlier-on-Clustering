from ucimlrepo import fetch_ucirepo
import pandas as pd
  
# Fetch dataset 
def fetch_dataset(): 
    return fetch_ucirepo(id=53)
  
# Get metadata
def get_metadata():
    return fetch_dataset().metadata

# Get variable information
def get_description():
    return fetch_dataset().variables

# Get data in original form
def get_data_numerical():
    iris = fetch_dataset()

    X = iris.data.features
    y = iris.data.targets

    return X, y

# Get data in categorical form
def get_data_categorical():
    X, y = get_data_numerical()

    for column in X.columns:
        X[column] = X[column].astype('category')

    return X, y
