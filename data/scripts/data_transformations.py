import pandas as pd

def split_df(df, number_of_columns=1):
    X = df.iloc[:, :-number_of_columns]
    y = df.iloc[:, -number_of_columns:]
    return X, y

def concat_df(df1, df2):
    return pd.concat([df1, df2], axis=0)