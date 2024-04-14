import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import adjusted_rand_score, confusion_matrix, roc_curve, auc

from data.scripts.get_data import get_data_numerical, get_data_categorical
from data.scripts.get_data_from_csv import get_data_from_csv, convert_iris_to_categorical
from data.scripts.data_transformations import split_df, concat_df
from data.scripts.plant_outliers import add_local_outliers, add_global_outliers, add_contextual_outliers, add_collective_outliers

################
# Data loading #
################

file_path_raw_data = '/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/data/numerical/Iris-czech.csv'
SPECIES_COLUMN_NAME = 'Druh'
IS_OUTLIER_COLUMN_NAME = 'IsOutlier'
df = get_data_from_csv(file_path_raw_data)
df = df.drop('Id', axis=1)
X, y = split_df(df)

NB_BINS = 7
RUN_NUMBER = 3
#########################
### Generate Outliers ###
# #########################
# np.random.seed(19)
# ### Local outliers
# df_with_local_outliers_1percent = add_local_outliers(df, outlier_percentage=1, rate=3.5, species_column=SPECIES_COLUMN_NAME)
# df_with_local_outliers_5percent = add_local_outliers(df, outlier_percentage=5, rate=3.5, species_column=SPECIES_COLUMN_NAME)
# df_with_local_outliers_10percent = add_local_outliers(df, outlier_percentage=10, rate=3.5, species_column=SPECIES_COLUMN_NAME)

# X_local_outliers_1percent, y_local_outliers_1percent = split_df(df_with_local_outliers_1percent, number_of_columns=2)
# X_local_outliers_5percent, y_local_outliers_5percent = split_df(df_with_local_outliers_5percent, number_of_columns=2)
# X_local_outliers_10percent, y_local_outliers_10percent = split_df(df_with_local_outliers_10percent, number_of_columns=2)

# ### Global outliers
# df_with_global_outliers_1percent = add_global_outliers(df, outlier_percentage=1, rate=3.5, species_column=SPECIES_COLUMN_NAME)
# df_with_global_outliers_5percent = add_global_outliers(df, outlier_percentage=5, rate=3.5, species_column=SPECIES_COLUMN_NAME)
# df_with_global_outliers_10percent = add_global_outliers(df, outlier_percentage=10, rate=3.5, species_column=SPECIES_COLUMN_NAME)

# X_global_outliers_1percent, y_global_outliers_1percent = split_df(df_with_global_outliers_1percent, number_of_columns=2)
# X_global_outliers_5percent, y_global_outliers_5percent = split_df(df_with_global_outliers_5percent, number_of_columns=2)
# X_global_outliers_10percent, y_global_outliers_10percent = split_df(df_with_global_outliers_10percent, number_of_columns=2)

# ### Contextual outliers
# df_with_contextual_outliers_1percent = add_contextual_outliers(df, outlier_percentage=1, num_columns=3, species_column=SPECIES_COLUMN_NAME)
# df_with_contextual_outliers_5percent = add_contextual_outliers(df, outlier_percentage=5, num_columns=3, species_column=SPECIES_COLUMN_NAME)
# df_with_contextual_outliers_10percent = add_contextual_outliers(df, outlier_percentage=10, num_columns=3, species_column=SPECIES_COLUMN_NAME)

# X_contextual_outliers_1percent, y_contextual_outliers_1percent = split_df(df_with_contextual_outliers_1percent, number_of_columns=2)
# X_contextual_outliers_5percent, y_contextual_outliers_5percent = split_df(df_with_contextual_outliers_5percent, number_of_columns=2)
# X_contextual_outliers_10percent, y_contextual_outliers_10percent = split_df(df_with_contextual_outliers_10percent, number_of_columns=2)

# ### Collective outliers
# df_with_collective_outliers_1percent = add_collective_outliers(df, 1, species_column=SPECIES_COLUMN_NAME)
# df_with_collective_outliers_5percent = add_collective_outliers(df, 5, species_column=SPECIES_COLUMN_NAME)
# df_with_collective_outliers_10percent = add_collective_outliers(df, 10, species_column=SPECIES_COLUMN_NAME)

# X_collective_outliers_1percent, y_collective_outliers_1percent = split_df(df_with_collective_outliers_1percent, number_of_columns=2)
# X_collective_outliers_5percent, y_collective_outliers_5percent = split_df(df_with_collective_outliers_5percent, number_of_columns=2)
# X_collective_outliers_10percent, y_collective_outliers_10percent = split_df(df_with_collective_outliers_10percent, number_of_columns=2)
np.random.seed(1911)

outlier_functions = [add_local_outliers, add_global_outliers, add_contextual_outliers, add_collective_outliers]
outlier_names = ['local', 'global', 'contextual', 'collective']
outlier_percentages = [1, 5, 10]

data = {}

for func, name in zip(outlier_functions, outlier_names):
    for percentage in outlier_percentages:
        if name == 'contextual':
            df_with_outliers = func(df, outlier_percentage=percentage, num_columns=2, species_column=SPECIES_COLUMN_NAME)
        elif name == 'collective':
            df_with_outliers = func(df, percentage, species_column=SPECIES_COLUMN_NAME)
        else:
            df_with_outliers = func(df, outlier_percentage=percentage, rate=3.5, species_column=SPECIES_COLUMN_NAME)
        
        X, y = split_df(df_with_outliers, number_of_columns=2)
        
        data[f'df_{name}_outliers_{percentage}percent'] = df_with_outliers
        # data[f'X_{name}_outliers_{percentage}percent'] = X
        # data[f'y_{name}_outliers_{percentage}percent'] = y

# print(df_with_local_outliers.head(5))
# print(df_with_local_outliers_10percent.tail(15))
# print(df_with_global_outliers_5percent.tail(15))
# print(df_with_contextual_outliers_1percent.tail(15))
# print(df_with_collective_outliers_10percent.tail(15))
# print(y_local_outliers_1percent)

for name, df in data.items():
    df.to_csv(f'data/numerical/wOutliers/run{RUN_NUMBER}/{name}.csv', index=False)
# print(data)


for name, df in data.items():
    for column in df.columns[:-2]:
        df[column] = pd.cut(df[column], bins=NB_BINS)
        df.to_csv(f'data/categorical/wOutliers/run{RUN_NUMBER}/{name}.csv', index=False)