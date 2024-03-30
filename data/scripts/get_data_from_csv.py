import pandas as pd

# Get data in original form
def get_data_from_csv(file_path):
    return pd.read_csv(file_path)

# Get data in categorical form
def convert_iris_to_categorical(file_path):
    df = get_data_from_csv(file_path)

    labels = ['a', 'b', 'c', 'd', 'e']
    for column in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:
        df[column] = pd.cut(df[column], bins=5, labels=labels)

    return df
