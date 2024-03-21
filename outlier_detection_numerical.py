import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from data.get_data import get_data_numerical, get_data_categorical
from data.get_data_from_csv import get_data_from_csv, convert_iris_to_categorical
from data_transformations import split_df, concat_df


####################
# Data preparation #
####################
file_path_raw_data = '/Users/ngocphuong.vu/skola/diplomka/code/Influence of Outliers on Clustering/data/Iris.csv'
file_path_outliers = '/Users/ngocphuong.vu/skola/diplomka/code/Influence of Outliers on Clustering/data/Iris-artificial-outliers.csv'

df_raw = get_data_from_csv(file_path_raw_data)
df_outliers = get_data_from_csv(file_path_outliers)

df = concat_df(df_raw, df_outliers)
X, y = split_df(df)

###################################
# Cluster based Outlier Detection #
###################################
###
### K-means
###
k = 9
percentage = 0.04

kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
clusters = kmeans.predict(X)

# Calculate the distance of each point to its cluster center
X_dist = kmeans.transform(X)
# Select the minimum distance to a cluster center as the representative distance
X_dist = np.min(X_dist, axis=1)

# Determine a threshold for outliers or use a top percentage
threshold = np.percentile(X_dist, 100 - percentage*100)
outliers_kMeans = X_dist > threshold

###
### DBScan
###
dbscan = DBSCAN(eps=2.8, min_samples=4) # Trial and error

# Fit the model
dbscan.fit(X)

# Outliers labelled with -1, different clusters get non-negative integers
outliers_DBScan = dbscan.labels_ == -1

####################
# Isolation Forest #
####################
iForest = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42)
iForest.fit(X)
outliers_iForest = iForest.predict(X)

########################
# Local Outlier Factor #
########################
LOF = LocalOutlierFactor(
    n_neighbors=7,
    contamination=0.02)
outliers_LOF = LOF.fit_predict(X)


# print out results
print("kMeans: ", outliers_kMeans, "\nDBScan: ", outliers_DBScan, "\niForest: ", outliers_iForest, "\nLOF: ", outliers_LOF)
