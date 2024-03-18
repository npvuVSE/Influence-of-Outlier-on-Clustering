from get_data import get_data_numerical, get_data_categorical
from get_data_from_csv import get_data_from_csv_numerical, get_data_from_csv_categorical
from data_transformations import split_df, concat_df

from sklearn.ensemble import IsolationForest

file_path_raw_data = '/Users/ngocphuong.vu/skola/diplomka/code/Influence of Outliers on Clustering/data/Iris.csv'
file_path_outliers = '/Users/ngocphuong.vu/skola/diplomka/code/Influence of Outliers on Clustering/data/Iris-artificial-outliers.csv'

df_raw = get_data_from_csv_numerical(file_path_raw_data)
df_outliers = get_data_from_csv_numerical(file_path_outliers)

df = concat_df(df_raw, df_outliers)
X, y = split_df(df)

print(X)

iForest = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42)
iForest.fit(X)

outliers = iForest.predict(X)
print(outliers)
