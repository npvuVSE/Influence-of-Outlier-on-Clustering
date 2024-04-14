import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import adjusted_rand_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform

from data.scripts.get_data import get_data_numerical
from data.scripts.get_data_from_csv import get_data_from_csv
from data.scripts.data_transformations import split_df, concat_df
from sklearn.metrics import silhouette_score, davies_bouldin_score

file_path_raw_data = '/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/data/numerical/Iris-czech.csv'
SPECIES_COLUMN_NAME = 'Druh'

df = get_data_from_csv(file_path_raw_data)
df = df.drop('Id', axis=1)
X, y = split_df(df)

# print(X.head())
# print(y.head()) 

n_cluster = 3
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# print(f'y: {y}')
# print(f'y_encoded: {y_encoded}')

# Hierarchical Clustering
linkages = ['single', 'average',  'complete', 'ward']
metrics = ['euclidean',] # 'manhattan']

for linkage_type in linkages:
    for metric_type in metrics:
        if linkage_type == 'ward' and metric_type != 'euclidean':
            continue
        
        model_agg = AgglomerativeClustering(n_clusters=n_cluster, metric=metric_type, linkage=linkage_type)
        labels_agg = model_agg.fit_predict(X)
        silhouette_index_agg = silhouette_score(X, labels_agg, metric=metric_type)
        db_index_agg = davies_bouldin_score(X, labels_agg)
        print(f'Agglomerative Clustering - Linkage: {linkage_type.title()}, \
              Metric: {metric_type.title()}, \
              Silhouette Index: {silhouette_index_agg:.4f}, \
              Davies-Bouldin Index: {db_index_agg:.4f}')

# KMeans
model_kmeans = KMeans(n_clusters=n_cluster)
labels_kmeans = model_kmeans.fit_predict(X)
silhouette_index_kmeans = silhouette_score(X, labels_kmeans)
db_index_kmeans = davies_bouldin_score(X, labels_kmeans)
print(f'KMeans - Silhouette Index: {silhouette_index_kmeans}, Davies-Bouldin Index: {db_index_kmeans}')        

######################
# Data with outliers #
######################

outlier_names = ['local', 'global', 'contextual', 'collective']
outlier_percentages = [1, 5, 10]

data_wOutliers = {}

for name in outlier_names:
    for percentage in outlier_percentages:
        df = pd.read_csv(f'data/numerical/wOutliers/run1/df_{name}_outliers_{percentage}percent.csv')
        data_wOutliers[f'df_{name}_outliers_{percentage}percent'] = df
    
# for name, df in data_wOutliers.items():
#     if 'local' not in name:
#         continue
#     X = df.drop(['Druh', 'IsOutlier'], axis=1)
    
#     for linkage_type in linkages:
#         for metric_type in metrics:
#             if linkage_type == 'ward' and metric_type != 'euclidean':
#                 continue

#             model_agg = AgglomerativeClustering(n_clusters=n_cluster, metric=metric_type, linkage=linkage_type)
#             labels_agg = model_agg.fit_predict(X)
#             silhouette_index_agg = silhouette_score(X, labels_agg, metric=metric_type)
#             db_index_agg = davies_bouldin_score(X, labels_agg)
#             print(f'DataFrame: {name}, Agglomerative Clustering - Linkage: {linkage_type.title()}, \
#                   Metric: {metric_type.title()}, \
#                   Silhouette Index: {silhouette_index_agg}, \
#                   Davies-Bouldin Index: {db_index_agg}')

#     model_kmeans = KMeans(n_clusters=n_cluster)
#     labels_kmeans = model_kmeans.fit_predict(X)
#     silhouette_index_kmeans = silhouette_score(X, labels_kmeans)
#     db_index_kmeans = davies_bouldin_score(X, labels_kmeans)
#     print(f'DataFrame: {name}, KMeans - Silhouette Index: {silhouette_index_kmeans}, Davies-Bouldin Index: {db_index_kmeans}')

# print for latex
linkage_names = {
    'single': '\\nearetsNeighbourMethod', 'average': '\\averageLinkageMethod', 
    'complete': '\\farthestNeighbourMethod', 'ward': '\\WardMethod'
    }

for name, df in data_wOutliers.items():
    if 'collective' not in name:
        continue

    X = df.drop(['Druh', 'IsOutlier'], axis=1)
    
    for linkage_type in linkages:
        for metric_type in metrics:
            if linkage_type == 'ward' and metric_type != 'euclidean':
                continue

            model_agg = AgglomerativeClustering(n_clusters=n_cluster, metric=metric_type, linkage=linkage_type)
            labels_agg = model_agg.fit_predict(X)
            silhouette_index_agg = silhouette_score(X, labels_agg, metric=metric_type)
            db_index_agg = davies_bouldin_score(X, labels_agg)
            print(f'{name.split("_")[-1]}\\% & {linkage_names[linkage_type]} & {silhouette_index_agg:.4f} & {db_index_agg:.4f} \\\\')

            # dist_matrix = squareform(pdist(X, metric=metric_type))
            # silhouette_index_agg = silhouette_score(dist_matrix, labels_agg, metric='precomputed')
            # db_index_agg = davies_bouldin_score(dist_matrix, labels_agg)
            # print(f'{name.split("_")[-1]}\\% & {linkage_names[linkage_type]} & {silhouette_index_agg:.4f} & {db_index_agg:.4f} \\\\')

    model_kmeans = KMeans(n_clusters=n_cluster)
    labels_kmeans = model_kmeans.fit_predict(X)
    silhouette_index_kmeans = silhouette_score(X, labels_kmeans)
    db_index_kmeans = davies_bouldin_score(X, labels_kmeans)
    print(f'{name.split("_")[-1]}\\% & \\kMeansMethod & {silhouette_index_kmeans:.4f} & {db_index_kmeans:.4f} \\\\')
