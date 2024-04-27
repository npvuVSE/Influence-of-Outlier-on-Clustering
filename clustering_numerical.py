import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import adjusted_rand_score, confusion_matrix, roc_curve, auc, adjusted_rand_score, silhouette_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform
import json

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
RANDOM_STATE = 1917
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
        ari_agg = adjusted_rand_score(y_encoded, labels_agg)
        print(f'Agglomerative Clustering - Linkage: {linkage_type.title()}, \
              Metric: {metric_type.title()}, \
              Silhouette Index: {silhouette_index_agg:.4f}, \
              Davies-Bouldin Index: {db_index_agg:.4f}, \
              Adjusted Rand Index: {ari_agg:.4f}')
        # print(f'Agglomerative Clustering - Linkage: {linkage_type.title()} - ARI: {ari_agg:.4f}')

# KMeans
model_kmeans = KMeans(n_clusters=n_cluster, random_state=RANDOM_STATE)
labels_kmeans = model_kmeans.fit_predict(X)
silhouette_index_kmeans = silhouette_score(X, labels_kmeans)
db_index_kmeans = davies_bouldin_score(X, labels_kmeans)
ari_kmeans = adjusted_rand_score(y_encoded, labels_kmeans)
print(f'KMeans - Silhouette Index: {silhouette_index_kmeans:.4f}, Davies-Bouldin Index: {db_index_kmeans:.4f}, Adjusted Rand Index: {ari_kmeans:.4f}')
# print(f'KMeans - ARI: {ari_kmeans:.4f}')

######################
# Data with outliers #
######################
local_1percent_nearest = []
local_5percent_nearest = []
local_10percent_nearest = []

global_1percent_nearest = []
global_5percent_nearest = []
global_10percent_nearest = []

contextual_1percent_nearest = []
contextual_5percent_nearest = []
contextual_10percent_nearest = []

collective_1percent_nearest = []
collective_5percent_nearest = []
collective_10percent_nearest = []

local_1percent_average = []
local_5percent_average = []
local_10percent_average = []

global_1percent_average = []
global_5percent_average = []
global_10percent_average = []

contextual_1percent_average = []
contextual_5percent_average = []
contextual_10percent_average = []

collective_1percent_average = []
collective_5percent_average = []
collective_10percent_average = []

local_1percent_farthest = []
local_5percent_farthest = []
local_10percent_farthest = []

global_1percent_farthest = []
global_5percent_farthest = []
global_10percent_farthest = []

contextual_1percent_farthest = []
contextual_5percent_farthest = []
contextual_10percent_farthest = []

collective_1percent_farthest = []
collective_5percent_farthest = []
collective_10percent_farthest = []

local_1percent_ward = []
local_5percent_ward = []
local_10percent_ward = []

global_1percent_ward = []
global_5percent_ward = []
global_10percent_ward = []

contextual_1percent_ward = []
contextual_5percent_ward = []
contextual_10percent_ward = []

collective_1percent_ward = []
collective_5percent_ward = []
collective_10percent_ward = []

local_1percent_kmeans = []
local_5percent_kmeans = []
local_10percent_kmeans = []

global_1percent_kmeans = []
global_5percent_kmeans = []
global_10percent_kmeans = []

contextual_1percent_kmeans = []
contextual_5percent_kmeans = []
contextual_10percent_kmeans = []

collective_1percent_kmeans = []
collective_5percent_kmeans = []
collective_10percent_kmeans = []

outlier_names = ['local', 'global', 'contextual', 'collective']
outlier_percentages = [1, 5, 10]

data_wOutliers = {}
metric_type = 'euclidean'
linkages = ['single', 'average',  'complete', 'ward']
for run_number in range(1, 51):
    for name in outlier_names:
        for percentage in outlier_percentages:
            df = pd.read_csv(f'/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/data/numerical/wOutliers/run{run_number}/df_{name}_outliers_{percentage}percent.csv')
            data_wOutliers[f'df_{name}_outliers_{percentage}percent'] = df

    for name, df in data_wOutliers.items():
        X = df.drop(['Druh', 'IsOutlier'], axis=1)
        # print(f'Processing {name}')
        nn_model = AgglomerativeClustering(n_clusters=n_cluster, metric=metric_type, linkage='single')
        nn_labels = nn_model.fit_predict(X)
        ave_model = AgglomerativeClustering(n_clusters=n_cluster, metric=metric_type, linkage='average')
        ave_labels = ave_model.fit_predict(X)
        fn_model = AgglomerativeClustering(n_clusters=n_cluster, metric=metric_type, linkage='complete')
        fn_label = fn_model.fit_predict(X)
        ward_model = AgglomerativeClustering(n_clusters=n_cluster, metric=metric_type, linkage='ward')
        ward_labels = ward_model.fit_predict(X)
        kmeans_model = KMeans(n_clusters=n_cluster)
        kmeans_labels = kmeans_model.fit_predict(X)

        si_nn = silhouette_score(X, nn_labels, metric=metric_type)
        si_ave = silhouette_score(X, ave_labels, metric=metric_type)
        si_fn = silhouette_score(X, fn_label, metric=metric_type)
        si_ward = silhouette_score(X, ward_labels, metric=metric_type)
        si_kmeans = silhouette_score(X, kmeans_labels)

        db_nn = davies_bouldin_score(X, nn_labels)
        db_ave = davies_bouldin_score(X, ave_labels)
        db_fn = davies_bouldin_score(X, fn_label)
        db_ward = davies_bouldin_score(X, ward_labels)
        db_kmeans = davies_bouldin_score(X, kmeans_labels)

        ari_nn = adjusted_rand_score(y_encoded, nn_labels[:len(y_encoded)])
        ari_ave = adjusted_rand_score(y_encoded, ave_labels[:len(y_encoded)])
        ari_fn = adjusted_rand_score(y_encoded, fn_label[:len(y_encoded)])
        ari_ward = adjusted_rand_score(y_encoded, ward_labels[:len(y_encoded)])
        ari_kmeans = adjusted_rand_score(y_encoded, kmeans_labels[:len(y_encoded)])

        if name == 'df_local_outliers_1percent':
            local_1percent_nearest.append([si_nn, db_nn, ari_nn])
            local_1percent_average.append([si_ave, db_ave, ari_ave])
            local_1percent_farthest.append([si_fn, db_fn, ari_fn])
            local_1percent_ward.append([si_ward, db_ward, ari_ward])
            local_1percent_kmeans.append([si_kmeans, db_kmeans, ari_kmeans])
        elif name == 'df_local_outliers_5percent':
            local_5percent_nearest.append([si_nn, db_nn, ari_nn])
            local_5percent_average.append([si_ave, db_ave, ari_ave])
            local_5percent_farthest.append([si_fn, db_fn, ari_fn])
            local_5percent_ward.append([si_ward, db_ward, ari_ward])
            local_5percent_kmeans.append([si_kmeans, db_kmeans, ari_kmeans])
        elif name == 'df_local_outliers_10percent':
            local_10percent_nearest.append([si_nn, db_nn, ari_nn])
            local_10percent_average.append([si_ave, db_ave, ari_ave])
            local_10percent_farthest.append([si_fn, db_fn, ari_fn])
            local_10percent_ward.append([si_ward, db_ward, ari_ward])
            local_10percent_kmeans.append([si_kmeans, db_kmeans, ari_kmeans])
        elif name == 'df_global_outliers_1percent':
            global_1percent_nearest.append([si_nn, db_nn, ari_nn])
            global_1percent_average.append([si_ave, db_ave, ari_ave])
            global_1percent_farthest.append([si_fn, db_fn, ari_fn])
            global_1percent_ward.append([si_ward, db_ward, ari_ward])
            global_1percent_kmeans.append([si_kmeans, db_kmeans, ari_kmeans])
        elif name == 'df_global_outliers_5percent':
            global_5percent_nearest.append([si_nn, db_nn, ari_nn])
            global_5percent_average.append([si_ave, db_ave, ari_ave])
            global_5percent_farthest.append([si_fn, db_fn, ari_fn])
            global_5percent_ward.append([si_ward, db_ward, ari_ward])
            global_5percent_kmeans.append([si_kmeans, db_kmeans, ari_kmeans])
        elif name == 'df_global_outliers_10percent':
            global_10percent_nearest.append([si_nn, db_nn, ari_nn])
            global_10percent_average.append([si_ave, db_ave, ari_ave])
            global_10percent_farthest.append([si_fn, db_fn, ari_fn])
            global_10percent_ward.append([si_ward, db_ward, ari_ward])
            global_10percent_kmeans.append([si_kmeans, db_kmeans, ari_kmeans])
        elif name == 'df_contextual_outliers_1percent':
            contextual_1percent_nearest.append([si_nn, db_nn, ari_nn])
            contextual_1percent_average.append([si_ave, db_ave, ari_ave])
            contextual_1percent_farthest.append([si_fn, db_fn, ari_fn])
            contextual_1percent_ward.append([si_ward, db_ward, ari_ward])
            contextual_1percent_kmeans.append([si_kmeans, db_kmeans, ari_kmeans])
        elif name == 'df_contextual_outliers_5percent':
            contextual_5percent_nearest.append([si_nn, db_nn, ari_nn])
            contextual_5percent_average.append([si_ave, db_ave, ari_ave])
            contextual_5percent_farthest.append([si_fn, db_fn, ari_fn])
            contextual_5percent_ward.append([si_ward, db_ward, ari_ward])
            contextual_5percent_kmeans.append([si_kmeans, db_kmeans, ari_kmeans])
        elif name == 'df_contextual_outliers_10percent':
            contextual_10percent_nearest.append([si_nn, db_nn, ari_nn])
            contextual_10percent_average.append([si_ave, db_ave, ari_ave])
            contextual_10percent_farthest.append([si_fn, db_fn, ari_fn])
            contextual_10percent_ward.append([si_ward, db_ward, ari_ward])
            contextual_10percent_kmeans.append([si_kmeans, db_kmeans, ari_kmeans])
        elif name == 'df_collective_outliers_1percent':
            collective_1percent_nearest.append([si_nn, db_nn, ari_nn])
            collective_1percent_average.append([si_ave, db_ave, ari_ave])
            collective_1percent_farthest.append([si_fn, db_fn, ari_fn])
            collective_1percent_ward.append([si_ward, db_ward, ari_ward])
            collective_1percent_kmeans.append([si_kmeans, db_kmeans, ari_kmeans])
        elif name == 'df_collective_outliers_5percent':
            collective_5percent_nearest.append([si_nn, db_nn, ari_nn])
            collective_5percent_average.append([si_ave, db_ave, ari_ave])
            collective_5percent_farthest.append([si_fn, db_fn, ari_fn])
            collective_5percent_ward.append([si_ward, db_ward, ari_ward])
            collective_5percent_kmeans.append([si_kmeans, db_kmeans, ari_kmeans])
        elif name == 'df_collective_outliers_10percent':
            collective_10percent_nearest.append([si_nn, db_nn, ari_nn])
            collective_10percent_average.append([si_ave, db_ave, ari_ave])
            collective_10percent_farthest.append([si_fn, db_fn, ari_fn])
            collective_10percent_ward.append([si_ward, db_ward, ari_ward])
            collective_10percent_kmeans.append([si_kmeans, db_kmeans, ari_kmeans])
        
print('Local 1% nearest')
print(np.mean(np.array(local_1percent_nearest), axis=0))

print('Local 5% nearest')
print(np.mean(np.array(local_5percent_nearest), axis=0))

print('Local 10% nearest')
print(np.mean(np.array(local_10percent_nearest), axis=0))

print('Global 1% nearest')
print(np.mean(np.array(global_1percent_nearest), axis=0))

print('Global 5% nearest')
print(np.mean(np.array(global_5percent_nearest), axis=0))

print('Global 10% nearest')
print(np.mean(np.array(global_10percent_nearest), axis=0))

print('Contextual 1% nearest')
print(np.mean(np.array(contextual_1percent_nearest), axis=0))

print('Contextual 5% nearest')
print(np.mean(np.array(contextual_5percent_nearest), axis=0))

print('Contextual 10% nearest')
print(np.mean(np.array(contextual_10percent_nearest), axis=0))

print('Collective 1% nearest')
print(np.mean(np.array(collective_1percent_nearest), axis=0))

print('Collective 5% nearest')
print(np.mean(np.array(collective_5percent_nearest), axis=0))

print('Collective 10% nearest')
print(np.mean(np.array(collective_10percent_nearest), axis=0))

print('Local 1% average')
print(np.mean(np.array(local_1percent_average), axis=0))

print('Local 5% average')
print(np.mean(np.array(local_5percent_average), axis=0))

print('Local 10% average')
print(np.mean(np.array(local_10percent_average), axis=0))

print('Global 1% average')
print(np.mean(np.array(global_1percent_average), axis=0))

print('Global 5% average')
print(np.mean(np.array(global_5percent_average), axis=0))

print('Global 10% average')
print(np.mean(np.array(global_10percent_average), axis=0))

print('Contextual 1% average')
print(np.mean(np.array(contextual_1percent_average), axis=0))

print('Contextual 5% average')
print(np.mean(np.array(contextual_5percent_average), axis=0))

print('Contextual 10% average')
print(np.mean(np.array(contextual_10percent_average), axis=0))

print('Collective 1% average')
print(np.mean(np.array(collective_1percent_average), axis=0))

print('Collective 5% average')
print(np.mean(np.array(collective_5percent_average), axis=0))

print('Collective 10% average')
print(np.mean(np.array(collective_10percent_average), axis=0))

print('Local 1% farthest')
print(np.mean(np.array(local_1percent_farthest), axis=0))

print('Local 5% farthest')
print(np.mean(np.array(local_5percent_farthest), axis=0))

print('Local 10% farthest')
print(np.mean(np.array(local_10percent_farthest), axis=0))

print('Global 1% farthest')
print(np.mean(np.array(global_1percent_farthest), axis=0))

print('Global 5% farthest')
print(np.mean(np.array(global_5percent_farthest), axis=0))

print('Global 10% farthest')
print(np.mean(np.array(global_10percent_farthest), axis=0))

print('Contextual 1% farthest')
print(np.mean(np.array(contextual_1percent_farthest), axis=0))

print('Contextual 5% farthest')
print(np.mean(np.array(contextual_5percent_farthest), axis=0))

print('Contextual 10% farthest')
print(np.mean(np.array(contextual_10percent_farthest), axis=0))

print('Collective 1% farthest')
print(np.mean(np.array(collective_1percent_farthest), axis=0))

print('Collective 5% farthest')
print(np.mean(np.array(collective_5percent_farthest), axis=0))

print('Collective 10% farthest')
print(np.mean(np.array(collective_10percent_farthest), axis=0))

print('Local 1% ward')
print(np.mean(np.array(local_1percent_ward), axis=0))

print('Local 5% ward')
print(np.mean(np.array(local_5percent_ward), axis=0))

print('Local 10% ward')
print(np.mean(np.array(local_10percent_ward), axis=0))

print('Global 1% ward')
print(np.mean(np.array(global_1percent_ward), axis=0))

print('Global 5% ward')
print(np.mean(np.array(global_5percent_ward), axis=0))

print('Global 10% ward')
print(np.mean(np.array(global_10percent_ward), axis=0))

print('Contextual 1% ward')
print(np.mean(np.array(contextual_1percent_ward), axis=0))

print('Contextual 5% ward')
print(np.mean(np.array(contextual_5percent_ward), axis=0))

print('Contextual 10% ward')
print(np.mean(np.array(contextual_10percent_ward), axis=0))

print('Collective 1% ward')
print(np.mean(np.array(collective_1percent_ward), axis=0))

print('Collective 5% ward')
print(np.mean(np.array(collective_5percent_ward), axis=0))

print('Collective 10% ward')
print(np.mean(np.array(collective_10percent_ward), axis=0))

print('Local 1% kmeans')
print(np.mean(np.array(local_1percent_kmeans), axis=0))

print('Local 5% kmeans')
print(np.mean(np.array(local_5percent_kmeans), axis=0))

print('Local 10% kmeans')
print(np.mean(np.array(local_10percent_kmeans), axis=0))

print('Global 1% kmeans')
print(np.mean(np.array(global_1percent_kmeans), axis=0))

print('Global 5% kmeans')
print(np.mean(np.array(global_5percent_kmeans), axis=0))

print('Global 10% kmeans')
print(np.mean(np.array(global_10percent_kmeans), axis=0))

print('Contextual 1% kmeans')
print(np.mean(np.array(contextual_1percent_kmeans), axis=0))

print('Contextual 5% kmeans')
print(np.mean(np.array(contextual_5percent_kmeans), axis=0))

print('Contextual 10% kmeans')
print(np.mean(np.array(contextual_10percent_kmeans), axis=0))

print('Collective 1% kmeans')
print(np.mean(np.array(collective_1percent_kmeans), axis=0))

print('Collective 5% kmeans')
print(np.mean(np.array(collective_5percent_kmeans), axis=0))

print('Collective 10% kmeans')
print(np.mean(np.array(collective_10percent_kmeans), axis=0))




# for name in outlier_names:
#     for percentage in outlier_percentages:
#         df = pd.read_csv(f'/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/data/numerical/wOutliers/run1/df_{name}_outliers_{percentage}percent.csv')
#         data_wOutliers[f'df_{name}_outliers_{percentage}percent'] = df

# # print for latex
# linkage_names = {
#     'single': '\\nearetsNeighbourMethod', 'average': '\\averageLinkageMethod', 
#     'complete': '\\farthestNeighbourMethod', 'ward': '\\WardMethod'
#     }

# for name, df in data_wOutliers.items():
#     # if name not in ['df_collective_outliers_1percent','df_collective_outliers_5percent', 'df_collective_outliers_10percent']:
#     #     continue
#     X = df.drop(['Druh', 'IsOutlier'], axis=1)
#     print(f'Processing {name}')
    
#     for linkage_type in linkages:
#         # if linkage_type != 'average':
#         #     continue
#         for metric_type in metrics:
#             if linkage_type == 'ward' and metric_type != 'euclidean':
#                 continue

#             model_agg = AgglomerativeClustering(n_clusters=n_cluster, metric=metric_type, linkage=linkage_type)
#             labels_agg = model_agg.fit_predict(X)
#             silhouette_index_agg = silhouette_score(X, labels_agg, metric=metric_type)
#             db_index_agg = davies_bouldin_score(X, labels_agg)
#             ari_agg = adjusted_rand_score(y_encoded, labels_agg[:len(y_encoded)])
#             # print(f'encoded: {y_encoded}')
#             # print(f'labels_agg: {labels_agg}')
#             print(f'{name.split("_")[-1]}\\% & {linkage_names[linkage_type]} & {silhouette_index_agg:.4f} & {db_index_agg:.4f} & {ari_agg:.4f} \\\\')
#             # print(f'{linkage_names[linkage_type]} ARI: {ari_agg:.4f}')

#     model_kmeans = KMeans(n_clusters=n_cluster, random_state=RANDOM_STATE)
#     labels_kmeans = model_kmeans.fit_predict(X)
#     silhouette_index_kmeans = silhouette_score(X, labels_kmeans)
#     db_index_kmeans = davies_bouldin_score(X, labels_kmeans)
#     ari_agg = adjusted_rand_score(y_encoded, labels_kmeans[:len(y_encoded)])
#     print(f'{name.split("_")[-1]}\\% & \\kMeansMethod & {silhouette_index_kmeans:.4f} & {db_index_kmeans:.4f} & {ari_agg:.4f} \\\\')
    # print(f'kMeansMethod ARI: {ari_agg:.4f}')

##############################
# Data with outliers removed #
##############################
outlier_detection_methods = ['KMeans', 'IsolationForest', 'LocalOutlierFactor']
data_wOutliers_removed_KMeans = {}
data_wOutliers_removed_LOF = {}
data_wOutliers_removed_iForest = {}

# for name in outlier_names:
#     for percentage in outlier_percentages:
#         for method_name in outlier_detection_methods:
#             df = pd.read_csv(f'/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/data/numerical/wOutliers/run1/removed/{method_name}/df_{name}_outliers_{percentage}percent_removed_{method_name}.csv')
#             if method_name == 'KMeans':
#                 data_wOutliers_removed_KMeans[f'df_{name}_outliers_{percentage}percent_removed_{method_name}'] = df
#             elif method_name == 'LocalOutlierFactor':
#                 data_wOutliers_removed_LOF[f'df_{name}_outliers_{percentage}percent_removed_{method_name}'] = df
#             elif method_name == 'IsolationForest':
#                 data_wOutliers_removed_iForest[f'df_{name}_outliers_{percentage}percent_removed_{method_name}'] = df

# # print(data_wOutliers_removed_KMeans.keys())
        
# # print for latex
# linkage_names = {
#     'single': '\\nearetsNeighbourMethod', 'average': '\\averageLinkageMethod',
#     'complete': '\\farthestNeighbourMethod', 'ward': '\\WardMethod'
#     }

# print('\n\nProcessing dataset with outliers removed by KMeans')
# for name, df in data_wOutliers_removed_KMeans.items():
#     X = df.drop(['Druh', 'IsOutlier'], axis=1)
#     print(f'\nProcessing {name}')
    
#     for linkage_type in linkages:
#         for metric_type in metrics:
#             if linkage_type == 'ward' and metric_type != 'euclidean':
#                 continue

#             model_agg = AgglomerativeClustering(n_clusters=n_cluster, metric=metric_type, linkage=linkage_type)
#             labels_agg = model_agg.fit_predict(X)
#             silhouette_index_agg = silhouette_score(X, labels_agg, metric=metric_type)
#             db_index_agg = davies_bouldin_score(X, labels_agg)
#             print(f'{name.split("_")[-1]}\\% & {linkage_names[linkage_type]} & {silhouette_index_agg:.4f} & {db_index_agg:.4f} \\\\')

#     model_kmeans = KMeans(n_clusters=n_cluster, random_state=RANDOM_STATE)
#     labels_kmeans = model_kmeans.fit_predict(X)
#     silhouette_index_kmeans = silhouette_score(X, labels_kmeans)
#     db_index_kmeans = davies_bouldin_score(X, labels_kmeans)
#     print(f'{name.split("_")[-1]}\\% & \\kMeansMethod & {silhouette_index_kmeans:.4f} & {db_index_kmeans:.4f} \\\\')

# print('\n\nProcessing dataset with outliers removed by Local Outlier Factor')
# for name, df in data_wOutliers_removed_LOF.items():
#     X = df.drop(['Druh', 'IsOutlier'], axis=1)
#     print(f'\nProcessing {name}')
    
#     for linkage_type in linkages:
#         for metric_type in metrics:
#             if linkage_type == 'ward' and metric_type != 'euclidean':
#                 continue

#             model_agg = AgglomerativeClustering(n_clusters=n_cluster, metric=metric_type, linkage=linkage_type)
#             labels_agg = model_agg.fit_predict(X)
#             silhouette_index_agg = silhouette_score(X, labels_agg, metric=metric_type)
#             db_index_agg = davies_bouldin_score(X, labels_agg)
#             print(f'{name.split("_")[-1]}\\% & {linkage_names[linkage_type]} & {silhouette_index_agg:.4f} \\\\') # & {db_index_agg:.4f} \\\\')

#     model_kmeans = KMeans(n_clusters=n_cluster, random_state=RANDOM_STATE)
#     labels_kmeans = model_kmeans.fit_predict(X)
#     silhouette_index_kmeans = silhouette_score(X, labels_kmeans)
#     db_index_kmeans = davies_bouldin_score(X, labels_kmeans)
#     print(f'{name.split("_")[-1]}\\% & \\kMeansMethod & {silhouette_index_kmeans:.4f} \\\\') # & {db_index_kmeans:.4f} \\\\')

# print('\n\nProcessing dataset with outliers removed by Isolation Forest')
# for name, df in data_wOutliers_removed_iForest.items():
#     X = df.drop(['Druh', 'IsOutlier'], axis=1)
#     print(f'\nProcessing {name}')
#     # print(X.head())
    
#     for linkage_type in linkages:
#         for metric_type in metrics:
#             model_agg = AgglomerativeClustering(n_clusters=n_cluster, metric=metric_type, linkage=linkage_type)
#             labels_agg = model_agg.fit_predict(X)
#             silhouette_index_agg = silhouette_score(X, labels_agg, metric=metric_type)
#             db_index_agg = davies_bouldin_score(X, labels_agg)
#             print(f'{name.split("_")[-1]}\\% & {linkage_names[linkage_type]} & {silhouette_index_agg:.4f} & {db_index_agg:.4f} \\\\')

#     model_kmeans = KMeans(n_clusters=n_cluster, random_state=RANDOM_STATE)
#     labels_kmeans = model_kmeans.fit_predict(X)
#     silhouette_index_kmeans = silhouette_score(X, labels_kmeans)
#     db_index_kmeans = davies_bouldin_score(X, labels_kmeans)
#     print(f'{name.split("_")[-1]}\\% & \\kMeansMethod & {silhouette_index_kmeans:.4f} & {db_index_kmeans:.4f} \\\\')


###################
# with Simulation #
###################

# outlier_names = ['local', 'global', 'contextual', 'collective']
# outlier_percentages = [1, 5, 10]
# outlier_detection_methods = ['KMeans', 'IsolationForest', 'LocalOutlierFactor']
# runs = range(1, 51)
# n_cluster = 3
# linkages = ['single', 'average', 'complete', 'ward']

# data_wOutliers_removed = {
#     method: {f'{name}_{percentage}': [] for name in outlier_names for percentage in outlier_percentages}
#     for method in outlier_detection_methods
# }

# for run in runs:
#     for name in outlier_names:
#         for percentage in outlier_percentages:
#             for method_name in outlier_detection_methods:
#                 filepath = f'/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/data/numerical/wOutliers/run{run}/removed/{method_name}/df_{name}_outliers_{percentage}percent_removed_{method_name}.csv'
#                 df = pd.read_csv(filepath)
#                 data_wOutliers_removed[method_name][f'{name}_{percentage}'].append(df)

# results = {
#     outlier_type: {
#         percentage: {
#             method: {
#                 'Agglomerative': {linkage: {'silhouette_scores': [], 'db_scores': []} for linkage in linkages},
#                 'KMeans': {'silhouette_scores': [], 'db_scores': []}
#             } for method in outlier_detection_methods
#         }
#         for percentage in outlier_percentages
#     }
#     for outlier_type in outlier_names
# }

# for method_name, datasets in data_wOutliers_removed.items():
#     for key, dfs in datasets.items():
#         outlier_type, str_percentage = key.split('_')
#         percentage = int(str_percentage)
#         for df in dfs:
#             X = df.drop(['Druh', 'IsOutlier'], axis=1)

#             for linkage in linkages:
#                 model_agg = AgglomerativeClustering(n_clusters=n_cluster, metric='euclidean', linkage=linkage)
#                 labels_agg = model_agg.fit_predict(X)
#                 silhouette = silhouette_score(X, labels_agg)
#                 db_index = davies_bouldin_score(X, labels_agg)

#                 results[outlier_type][percentage][method_name]['Agglomerative'][linkage]['silhouette_scores'].append(silhouette)
#                 results[outlier_type][percentage][method_name]['Agglomerative'][linkage]['db_scores'].append(db_index)
            
#             model_kmeans = KMeans(n_clusters=n_cluster)
#             labels_kmeans = model_kmeans.fit_predict(X)
#             silhouette_kmeans = silhouette_score(X, labels_kmeans)
#             db_index_kmeans = davies_bouldin_score(X, labels_kmeans)

#             results[outlier_type][percentage][method_name]['KMeans']['silhouette_scores'].append(silhouette_kmeans)
#             results[outlier_type][percentage][method_name]['KMeans']['db_scores'].append(db_index_kmeans)

# for outlier_type, types in results.items():
#     print(f"\nResults for Outlier Type: {outlier_type}")
#     for percentage, percent_data in types.items():
#         print(f"  Percentage: {percentage}%")
#         for method, method_data in percent_data.items():
#             print(f"  Method: {method}")
#             for clustering_method, data in method_data.items():
#                 if clustering_method == 'KMeans':
#                     avg_silhouette = np.mean(data['silhouette_scores'])
#                     avg_db = np.mean(data['db_scores'])
#                     # print(f"    {method} - KMeans: Avg Silhouette: {avg_silhouette:.4f} & {avg_db:.4f} \\\\")
#                     print(f" &   & \\kMeansMethod & {avg_silhouette:.4f} & {avg_db:.4f} \\\\")
#                 else:
#                     for linkage, scores in data.items():
#                         # if method == 'IsolationForest' and outlier_type == 'contextual':
#                             # print(f"{percentage}%, {clustering_method}: {scores['silhouette_scores']}")
#                         avg_silhouette = np.mean(scores['silhouette_scores'])
#                         avg_db = np.mean(scores['db_scores'])
#                         # print(f"    {method} - Agglomerative ({linkage}): Avg Silhouette: {avg_silhouette:.4f} & {avg_db:.4f} \\\\")
#                         if linkage == 'single':
#                             print(f" & & \\nearestNeighbourMethod &  {avg_silhouette:.4f} & {avg_db:.4f} \\\\")
#                         elif linkage == 'average':
#                             print(f" & & \\averageLinkageMethod &  {avg_silhouette:.4f} & {avg_db:.4f} \\\\")
#                         elif linkage == 'complete':
#                             print(f" & & \\farthestNeighbourMethod &  {avg_silhouette:.4f} & {avg_db:.4f} \\\\")
#                         elif linkage == 'ward':
#                             print(f" & & \\WardMethod &  {avg_silhouette:.4f} & {avg_db:.4f} \\\\")

# with open('/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/clustering_results.json', 'w') as f:
#     json.dump(results, f, indent=4)

####################################################################
# Plot difference between original and dataset with added outleirs #
####################################################################

# local outliers
local_outliers_ARI = [
    {
        'single': 0.5675,
        'average': 0.6135,
        'complete': 0.6061,
        'ward': 0.7311,
        'kmeans': 0.7106
    },
    {
        'single': 0.0454,
        'average': 0.5605,
        'complete': 0.5301,
        'ward': 0.7269,
        'kmeans': 0.7225
    },
    {
        'single': 0.0227,
        'average': 0.5507,
        'complete': 0.5048,
        'ward': 0.7248,
        'kmeans': 0.7183
    }
]

# global outliers
global_outliers_ARI = [
    {
        'single': 0.5681,
        'average': 0.5681,
        'complete': 0.4440,
        'ward': 0.7163,
        'kmeans': 0.7266
    },
    {
        'single': 0.0000,
        'average': 0.0227,
        'complete': 0.4640,
        'ward': 0.6747,
        'kmeans': 0.6451
    },
    {
        'single': 0.0000,
        'average': 0.0000,
        'complete': 0.3957,
        'ward': 0.5909,
        'kmeans': 0.5641
    }
]

# contextual outliers
contextual_outliers_ARI = [
    {
        'single': 0.5680,
        'average': 0.6250,
        'complete': 0.6340,
        'ward': 0.7311,
        'kmeans': 0.7058
    },
    {
        'single': 0.1931,
        'average': 0.5598,
        'complete': 0.6109,
        'ward': 0.7292,
        'kmeans': 0.7119
    },
    {
        'single': 0.0340,
        'average': 0.5555,
        'complete': 0.6100,
        'ward': 0.7314,
        'kmeans': 0.7135
    }
]

# collective outliers
collective_outliers_ARI = [
    {
        'single': 0.5681,
        'average': 0.5681,
        'complete': 0.6334,
        'ward': 0.7311,
        'kmeans': 0.7216
    },
    {
        'single': 0.5681,
        'average': 0.5681,
        'complete': 0.5981,
        'ward': 0.7311,
        'kmeans': 0.7045
    },
    {
        'single': 0.5681,
        'average': 0.5681,
        'complete': 0.5849,
        'ward': 0.5876,
        'kmeans': 0.6567
    }
]

initial_ARI_values = {
    'single': 0.5638,
    'average': 0.7592,
    'complete': 0.6423,
    'ward': 0.7312,
    'kmeans': 0.7163
}

name_mapping = {
    'single': 'nejbližšího souseda',
    'average': 'průměrné vazby',
    'complete': 'nejvzdálenějšího souseda',
    'ward': 'Wardova',
    'kmeans': 'k-průměrů'
}


def replace_names(data):
    return [{name_mapping[key]: value for key, value in item.items()} for item in data]

local_outliers_ARI = replace_names(local_outliers_ARI)
global_outliers_ARI = replace_names(global_outliers_ARI)
contextual_outliers_ARI = replace_names(contextual_outliers_ARI)
collective_outliers_ARI = replace_names(collective_outliers_ARI)
initial_ARI_values = replace_names([initial_ARI_values])[0]

outliers_ARI = {
    'lokální': local_outliers_ARI,
    'globální': global_outliers_ARI,
    'kontextuální': contextual_outliers_ARI,
    'kolektivní': collective_outliers_ARI
}

for outlier_type, outlier_data in outliers_ARI.items():
    data = []
    for i, next_set in enumerate(outlier_data):
        for method in initial_ARI_values:
            difference = initial_ARI_values[method] - next_set[method]
            if i == 0:
                data.append([f'{i+1}%', method, difference])
            else:
                data.append([f'{i*5}%', method, difference])
    df = pd.DataFrame(data, columns=['Percentage', 'Metoda', 'Difference'])
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Percentage', y='Difference', hue='Metoda', markers=True, dashes=False, style='Metoda')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f'Rozdíl v hodnotách ARI pro {outlier_type} odlehlé hodnoty', fontsize=16)
    plt.xlabel('Procento odlehlých hodnot', fontsize=14)
    plt.ylabel('Rozdíl v hodnotách ARI', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.grid(True)
    plt.tight_layout()

    # plt.savefig(f'plots/{outlier_type}_ari_diff_numerical.pdf', format='pdf')
    # plt.show()

######################

original_scores_local = {
    '1%': {
        'nejbližšího souseda': 0.5740,
        'průměrné vazby': 0.5831,
        'nejvzdálenějšího souseda': 0.5131,
        'Wardova': 0.5493,
        'k-průměrů': 0.5456
    },
    '5%': {
        'nejbližšího souseda': 0.2520,
        'průměrné vazby': 0.6131,
        'nejvzdálenějšího souseda': 0.5112,
        'Wardova': 0.5192,
        'k-průměrů': 0.5219
    },
    '10%': {
        'nejbližšího souseda': 0.2555,
        'průměrné vazby': 0.5866,
        'nejvzdálenějšího souseda': 0.4693,
        'Wardova': 0.4895,
        'k-průměrů': 0.4954
    }
}

original_scores_global = {
    '1%': {
        'nejbližšího souseda': 0.6688,
        'průměrné vazby': 0.6688,
        'nejvzdálenějšího souseda': 0.5024,
        'Wardova': 0.5398,
        'k-průměrů': 0.5458
    },
    '5%': {
        'nejbližšího souseda': 0.5178,
        'průměrné vazby': 0.5470,
        'nejvzdálenějšího souseda': 0.5551,
        'Wardova': 0.5544,
        'k-průměrů': 0.5739
    },
    '10%': {
        'nejbližšího souseda': 0.5307,
        'průměrné vazby': 0.5771,
        'nejvzdálenějšího souseda': 0.5311,
        'Wardova': 0.5968,
        'k-průměrů': 0.5882
    }
}

original_scores_contextual = {
    '1%': {
        'nejbližšího souseda': 0.5489,
        'průměrné vazby': 0.5685,
        'nejvzdálenějšího souseda': 0.5089,
        'Wardova': 0.5485,
        'k-průměrů': 0.5470
    },
    '5%': {
        'nejbližšího souseda': 0.2928,
        'průměrné vazby': 0.5985,
        'nejvzdálenějšího souseda': 0.4893,
        'Wardova': 0.5160,
        'k-průměrů': 0.5197
    },
    '10%': {
        'nejbližšího souseda': 0.1419,
        'průměrné vazby': 0.5808,
        'nejvzdálenějšího souseda': 0.4594,
        'Wardova': 0.4785,
        'k-průměrů': 0.4901
    }
}

original_scores_collective = {
    '1%': {
        'nejbližšího souseda': 0.6278,
        'průměrné vazby': 0.6278,
        'nejvzdálenějšího souseda': 0.5054,
        'Wardova': 0.5479,
        'k-průměrů': 0.5453
    },
    '5%': {
        'nejbližšího souseda': 0.6414,
        'průměrné vazby': 0.6414,
        'nejvzdálenějšího souseda': 0.4709,
        'Wardova': 0.5198,
        'k-průměrů': 0.5289
    },
    '10%': {
        'nejbližšího souseda': 0.6445,
        'průměrné vazby': 0.6445,
        'nejvzdálenějšího souseda': 0.4523,
        'Wardova': 0.6272,
        'k-průměrů': 0.5444
    }
}

removed_scores_local_kmeans = {
    '1%': {
        'nejbližšího souseda': 0.5088,
        'průměrné vazby': 0.5510,
        'nejvzdálenějšího souseda': 0.5145,
        'Wardova': 0.5546,
        'k-průměrů': 0.5522
    },
    '5%': {
        'nejbližšího souseda': 0.4908,
        'průměrné vazby': 0.5395,
        'nejvzdálenějšího souseda': 0.5188,
        'Wardova': 0.5535,
        'k-průměrů': 0.5522
    },
    '10%': {
        'nejbližšího souseda': 0.4589,
        'průměrné vazby': 0.5355,
        'nejvzdálenějšího souseda': 0.5269,
        'Wardova': 0.5499,
        'k-průměrů': 0.5477
    }
}

removed_scores_global_kmeans = {
    '1%': {
        'nejbližšího souseda': 0.5118,
        'průměrné vazby': 0.5539,
        'nejvzdálenějšího souseda': 0.5134,
        'Wardova': 0.5541,
        'k-průměrů': 0.5516
    },
    '5%': {
        'nejbližšího souseda': 0.5333,
        'průměrné vazby': 0.5693,
        'nejvzdálenějšího souseda': 0.5346,
        'Wardova': 0.5652,
        'k-průměrů': 0.5580
    },
    '10%': {
        'nejbližšího souseda': 0.5761,
        'průměrné vazby': 0.6093,
        'nejvzdálenějšího souseda': 0.5666,
        'Wardova': 0.5744,
        'k-průměrů': 0.5578
    }
}

removed_scores_contextual_kmeans = {
    '1%': {
        'nejbližšího souseda': 0.5076,
        'průměrné vazby': 0.5522,
        'nejvzdálenějšího souseda': 0.5146,
        'Wardova': 0.5544,
        'k-průměrů': 0.5511
    },
    '5%': {
        'nejbližšího souseda': 0.5197,
        'průměrné vazby': 0.5421,
        'nejvzdálenějšího souseda': 0.5227,
        'Wardova': 0.5555,
        'k-průměrů': 0.5525
    },
    '10%': {
        'nejbližšího souseda': 0.5110,
        'průměrné vazby': 0.5344,
        'nejvzdálenějšího souseda': 0.5314,
        'Wardova': 0.5528,
        'k-průměrů': 0.5502
    }
}

removed_scores_collective_kmeans = {
    '1%': {
        'nejbližšího souseda': 0.5118,
        'průměrné vazby': 0.5539,
        'nejvzdálenějšího souseda': 0.5134,
        'Wardova': 0.5541,
        'k-průměrů': 0.5502
    },
    '5%': {
        'nejbližšího souseda': 0.5118,
        'průměrné vazby': 0.5539,
        'nejvzdálenějšího souseda': 0.5134,
        'Wardova': 0.5541,
        'k-průměrů': 0.5509
    },
    '10%': {
        'nejbližšího souseda': 0.5572,
        'průměrné vazby': 0.5824,
        'nejvzdálenějšího souseda': 0.5119,
        'Wardova': 0.5527,
        'k-průměrů': 0.5496
    }
}

removed_scores_local_lof = {
    '1%': {
        'nejbližšího souseda': 0.5100,
        'průměrné vazby': 0.5505,
        'nejvzdálenějšího souseda': 0.5151,
        'Wardova': 0.5537,
        'k-průměrů': 0.5514
    },
    '5%': {
        'nejbližšího souseda': 0.5115,
        'průměrné vazby': 0.5474,
        'nejvzdálenějšího souseda': 0.5155,
        'Wardova': 0.5514,
        'k-průměrů': 0.5496
    },
    '10%': {
        'nejbližšího souseda': 0.5083,
        'průměrné vazby': 0.5472,
        'nejvzdálenějšího souseda': 0.5155,
        'Wardova': 0.5502,
        'k-průměrů': 0.5474
    }
}

removed_scores_global_lof = {
    '1%': {
        'nejbližšího souseda': 0.5118,
        'průměrné vazby': 0.5539,
        'nejvzdálenějšího souseda': 0.5134,
        'Wardova': 0.5541,
        'k-průměrů': 0.5508
    },
    '5%': {
        'nejbližšího souseda': 0.5118,
        'průměrné vazby': 0.5539,
        'nejvzdálenějšího souseda': 0.5134,
        'Wardova': 0.5541,
        'k-průměrů': 0.5508
    },
    '10%': {
        'nejbližšího souseda': 0.5585,
        'průměrné vazby': 0.5888,
        'nejvzdálenějšího souseda': 0.5128,
        'Wardova': 0.5558,
        'k-průměrů': 0.5553
    }
}

removed_scores_contextual_lof = {
    '1%': {
        'nejbližšího souseda': 0.5078,
        'průměrné vazby': 0.5514,
        'nejvzdálenějšího souseda': 0.5132,
        'Wardova': 0.5537,
        'k-průměrů': 0.5500
    },
    '5%': {
        'nejbližšího souseda': 0.5011,
        'průměrné vazby': 0.5436,
        'nejvzdálenějšího souseda': 0.5129,
        'Wardova': 0.5496,
        'k-průměrů': 0.5490
    },
    '10%': {
        'nejbližšího souseda': 0.5305,
        'průměrné vazby': 0.5682,
        'nejvzdálenějšího souseda': 0.5138,
        'Wardova': 0.5477,
        'k-průměrů': 0.5449
    }
}

removed_scores_collective_lof =  {
    '1%': {
        'nejbližšího souseda': 0.5118,
        'průměrné vazby': 0.5539,
        'nejvzdálenějšího souseda': 0.5134,
        'Wardova': 0.5541,
        'k-průměrů': 0.5517
    },
    '5%': {
        'nejbližšího souseda': 0.6521,
        'průměrné vazby': 0.6521,
        'nejvzdálenějšího souseda': 0.4741,
        'Wardova': 0.5247,
        'k-průměrů': 0.5444
    },
    '10%': {
        'nejbližšího souseda': 0.6756,
        'průměrné vazby': 0.6756,
        'nejvzdálenějšího souseda': 0.4995,
        'Wardova': 0.6691,
        'k-průměrů': 0.5746
    }
}

removed_scores_local_iForest = {
    '1%': {
        'nejbližšího souseda': 0.5164,
        'průměrné vazby': 0.5538,
        'nejvzdálenějšího souseda': 0.5215,
        'Wardova': 0.5530,
        'k-průměrů': 0.5498
    },
    '5%': {
        'nejbližšího souseda': 0.5298,
        'průměrné vazby': 0.5393,
        'nejvzdálenějšího souseda': 0.5259,
        'Wardova': 0.5429,
        'k-průměrů': 0.5359
    },
    '10%': {
        'nejbližšího souseda': 0.5170,
        'průměrné vazby': 0.5308,
        'nejvzdálenějšího souseda': 0.5225,
        'Wardova': 0.5333,
        'k-průměrů': 0.5256
    }
}

removed_scores_global_iForest = {
    '1%': {
        'nejbližšího souseda': 0.5118,
        'průměrné vazby': 0.5539,
        'nejvzdálenějšího souseda': 0.5134,
        'Wardova': 0.5541,
        'k-průměrů': 0.5516
    },
    '5%': {
        'nejbližšího souseda': 0.5118,
        'průměrné vazby': 0.5539,
        'nejvzdálenějšího souseda': 0.5134,
        'Wardova': 0.5541,
        'k-průměrů': 0.5516
    },
    '10%': {
        'nejbližšího souseda': 0.5118,
        'průměrné vazby': 0.5539,
        'nejvzdálenějšího souseda': 0.5134,
        'Wardova': 0.5541,
        'k-průměrů': 0.5518
    }
}

removed_scores_contextual_iForest = {
    '1%': {
        'nejbližšího souseda': 0.5056,
        'průměrné vazby': 0.5385,
        'nejvzdálenějšího souseda': 0.5197,
        'Wardova': 0.5516,
        'k-průměrů': 0.5492
    },
    '5%': {
        'nejbližšího souseda': 0.5355,
        'průměrné vazby': 0.5195,
        'nejvzdálenějšího souseda': 0.5301,
        'Wardova': 0.5403,
        'k-průměrů': 0.5367
    },
    '10%': {
        'nejbližšího souseda': 0.5054,
        'průměrné vazby': 0.4992,
        'nejvzdálenějšího souseda': 0.5251,
        'Wardova': 0.5311,
        'k-průměrů': 0.5206
    }
}

removed_scores_collective_iForest = {
    '1%': {
        'nejbližšího souseda': 0.5118,
        'průměrné vazby': 0.5539,
        'nejvzdálenějšího souseda': 0.5134,
        'Wardova': 0.5541,
        'k-průměrů': 0.5510
    },
    '5%': {
        'nejbližšího souseda': 0.6489,
        'průměrné vazby': 0.6431,
        'nejvzdálenějšího souseda': 0.5292,
        'Wardova': 0.5303,
        'k-průměrů': 0.5322
    },
    '10%': {
        'nejbližšího souseda': 0.6646,
        'průměrné vazby': 0.6566,
        'nejvzdálenějšího souseda': 0.5074,
        'Wardova': 0.5322,
        'k-průměrů': 0.5553
    }
}

removed_methods_local = {
    'k-průměrů': removed_scores_local_kmeans,
    'LOF': removed_scores_local_lof,
    'iForest': removed_scores_local_iForest
}

removed_methods_global = {
    'k-průměrů': removed_scores_global_kmeans,
    'LOF': removed_scores_global_lof,
    'iForest': removed_scores_global_iForest
}

removed_methods_contextual = {
    'k-průměrů': removed_scores_contextual_kmeans,
    'LOF': removed_scores_contextual_lof,
    'iForest': removed_scores_contextual_iForest
}

removed_methods_collective = {
    'k-průměrů': removed_scores_collective_kmeans,
    'LOF': removed_scores_collective_lof,
    'iForest': removed_scores_collective_iForest
}

data_local = []
percentages = ['1%', '5%', '10%']
for method, scores_dict in removed_methods_local.items():
    for percent, scores in scores_dict.items():
        for cluster_method, score in scores.items():
            difference =  score - original_scores_local[percent][cluster_method]
            data_local.append({
                'Metoda detekce': method,
                'Percentage': percent,
                'Metoda shlukování': cluster_method,
                'Difference in Silhouette Score': difference
            })
df_local = pd.DataFrame(data_local)

data_global = []
percentages = ['1%', '5%', '10%']
for method, scores_dict in removed_methods_global.items():
    for percent, scores in scores_dict.items():
        for cluster_method, score in scores.items():
            difference =  score - original_scores_global[percent][cluster_method]
            data_global.append({
                'Metoda detekce': method,
                'Percentage': percent,
                'Metoda shlukování': cluster_method,
                'Difference in Silhouette Score': difference
            })
df_global = pd.DataFrame(data_global)

data_contextual = []
percentages = ['1%', '5%', '10%']
for method, scores_dict in removed_methods_contextual.items():
    for percent, scores in scores_dict.items():
        for cluster_method, score in scores.items():
            difference =  score - original_scores_contextual[percent][cluster_method]
            data_contextual.append({
                'Metoda detekce': method,
                'Percentage': percent,
                'Metoda shlukování': cluster_method,
                'Difference in Silhouette Score': difference
            })
df_contextual = pd.DataFrame(data_contextual)

data_collective = []
percentages = ['1%', '5%', '10%']
for method, scores_dict in removed_methods_collective.items():
    for percent, scores in scores_dict.items():
        for cluster_method, score in scores.items():
            difference =  score - original_scores_collective[percent][cluster_method]
            data_collective.append({
                'Metoda detekce': method,
                'Percentage': percent,
                'Metoda shlukování': cluster_method,
                'Difference in Silhouette Score': difference
            })
df_collective = pd.DataFrame(data_collective)

dataframes = {
    'lokálních': df_local,
    'globálních': df_global,
    'kontextuálních': df_contextual,
    'kolektivních': df_collective
}

line_styles = {
    'k-průměrů': (3, 5),  
    'LOF': (1, 1),     
    'iForest': (),     
}

for df_name, df in dataframes.items():
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='Percentage', y='Difference in Silhouette Score', hue='Metoda shlukování', style='Metoda detekce', markers=True, dashes=line_styles)
    plt.title(f'Rozdíly ve skóre Silhouette po odstranění {df_name} odlehlých hodnot', loc='center', fontsize=16)
    plt.xlabel('Procento odstraněných odlehlých hodnot', fontsize=14)
    plt.ylabel('Rozdíl ve skóre Silhouette', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    # plt.grid(True)
    plt.tight_layout()

    plt.savefig(f'plots/{df_name}_silhouette_outliers_removed_diff_numerical.pdf', format='pdf')
    plt.show()
    
print(df_local.head())
print(df_global.head())