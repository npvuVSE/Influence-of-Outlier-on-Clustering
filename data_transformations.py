import pandas as pd

def split_df(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def concat_df(df1, df2):
    return pd.concat([df1, df2], axis=0)