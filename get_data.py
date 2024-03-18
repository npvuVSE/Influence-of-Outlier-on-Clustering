from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
def fetch_dataset(): 
    return fetch_ucirepo(id=53)
  
# get metadata
def get_metadata():
    return fetch_dataset().metadata

# get variable information 
def get_description():
    return fetch_dataset().variables

# get data in original form
def get_data_numerical():
    iris = fetch_dataset()

    X = iris.data.features 
    y = iris.data.targets 

    return X, y

# get data in categorical form
def get_data_categorical():
    X, y = get_data_numerical()

    for column in X.columns:
        X[column] = X[column].astype('category')

    return X, y
