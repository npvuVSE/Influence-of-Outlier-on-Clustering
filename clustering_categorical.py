import numpy as np
import pandas as pd
import gower
import seaborn as sns
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import adjusted_rand_score, confusion_matrix, roc_curve, auc
from scipy.spatial.distance import pdist, squareform, jaccard
from scipy.cluster.hierarchy import linkage, dendrogram


from data.scripts.get_data import get_data_numerical
from data.scripts.get_data_from_csv import get_data_from_csv
from data.scripts.data_transformations import split_df, concat_df
from sklearn.metrics import silhouette_score, davies_bouldin_score
from kmodes.kmodes import KModes

# def hamming_distance(u, v):
#     return np.sum(u != v) / len(u)

def smc_distance(u, v):
    matches = np.sum(u == v)
    total = len(u)
    similarity = matches / total
    return 1 - similarity

def calculate_dbi(smc_matrix, labels):
    distances = np.array(smc_matrix)
    k = len(np.unique(labels))

    centroids = []
    for i in range(k):
        indices = np.where(labels == i)[0]
        intra_distances = distances[np.ix_(indices, indices)]
        medoid_index = indices[np.argmin(intra_distances.sum(axis=1))]
        centroids.append(medoid_index)

    s = np.zeros(k)
    for i in range(k):
        indices = np.where(labels == i)[0]
        s[i] = np.mean(distances[centroids[i], indices])

    d = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d[i, j] = distances[centroids[i], centroids[j]]
            d[j, i] = d[i, j]

    R = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                R[i, j] = (s[i] + s[j]) / d[i, j]

    dbi = np.mean([np.max(R[i]) for i in range(k)])
    return dbi

file_path_raw_data = '/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/data/numerical/Iris-czech.csv'
SPECIES_COLUMN_NAME = 'Druh'
NB_BINS = 7
NB_CLUSTERS = 3
df = get_data_from_csv(file_path_raw_data)
df = df.drop('Id', axis=1)

df_cut = df.copy()

df_columns = df.columns
for column in df_columns[:-1]:
    df_cut[column] = pd.cut(df_cut[column], bins=NB_BINS)
    print(df_cut[column].value_counts())

# print(df_cut.dtypes)
# print(df_cut.head())
# print(df_qcut.head())

X_cut, y_cut = split_df(df_cut)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_cut)
# print(f'y_cut: {y_cut}, y_encoded: {y_encoded}')
# X_qcut, y_qcut = split_df(df_qcut)


kmodes_cut = KModes(n_clusters=NB_CLUSTERS, random_state=42)
clusters_cut = kmodes_cut.fit_predict(X_cut)

print(f'Clusters for cut data: {clusters_cut}')
# print(f'Clusters for qcut data: {clusters_qcut}')

X_cut = X_cut.astype(str)
print(X_cut.dtypes)

# def jaccard_similarity(x, y):
#     set_x = set(x)
#     set_y = set(y)
#     intersection = len(set_x & set_y)
#     union = len(set_x | set_y)
#     return intersection / union

# dist_matrix_jaccard_cut = squareform(pdist(X_cut, lambda u, v: 1 - jaccard_similarity(u, v)))
# dist_matrix_jaccard_qcut = squareform(pdist(X_qcut, lambda u, v: 1 - jaccard_similarity(u, v)))
# print(dist_matrix_jaccard_cut)
# dist_matrix_hamming_cut = squareform(pdist(X_cut, metric=hamming_distance))
dist_matrix_smc_cut = squareform(pdist(X_cut, metric=smc_distance))

# distance_matrix_gower_cut = gower.gower_matrix(X_cut)
# distance_matrix_gower_qcut = gower.gower_matrix(X_qcut)

print(f'SMC distance matrix cut: {dist_matrix_smc_cut}')
# f'Gower distance matrix cut: {distance_matrix_gower_cut}, \n \
    #   Hamming distance matrix cut: {dist_matrix_hamming_cut}, \n \
linkages = ['single', 'average', 'complete']
metrics = ['smc']
# dist_matrices = {
#     'gower': distance_matrix_gower_cut,
#     'hamming': dist_matrix_hamming_cut,
#     'smc': dist_matrix_smc_cut,
#     'jaccard': dist_matrix_jaccard_cut
# }
data = {}

for metric in metrics:
    for linkage in linkages:
        clusters = AgglomerativeClustering(n_clusters=NB_CLUSTERS, linkage=linkage, metric='precomputed').fit_predict(dist_matrix_smc_cut)
        data[f'Hierarchical {linkage.capitalize()} {metric.capitalize()}'] = (dist_matrix_smc_cut, clusters)
        # print(f'Clusters for hierarchical {linkage} cut {metric}: {clusters}')

    data[f'K-modes {metric.capitalize()}'] = (dist_matrix_smc_cut, clusters_cut)

for name, (dist_matrix, clusters) in data.items():
    silhouette_index = silhouette_score(dist_matrix, clusters, metric='precomputed')
    dbi_index = calculate_dbi(dist_matrix, clusters)
    ari_index = adjusted_rand_score(y_encoded, clusters)
    print(f'{name} & {silhouette_index:.4f} & {dbi_index:.4f} & {ari_index:.4f} \\\\')

#  hierarchical clustering
# Z = linkage(dist_matrix, 'average')
# print(Z)

#  dendrogram
# plt.figure(figsize=(8, 4))
# dendrogram(Z, labels=np.arange(1, len(X_cut) + 1))
# plt.title('Hierarchical Clustering with Jaccard Similarity')
# plt.xlabel('Data Points')
# plt.ylabel('Dissimilarity')
# plt.show()


######################
# Data with outliers #
######################

outlier_names = ['local', 'global', 'contextual', 'collective']
outlier_percentages = [1, 5, 10]

data_wOutliers = {}

for name in outlier_names:
    for percentage in outlier_percentages:
        df = pd.read_csv(f'data/categorical/wOutliers/run1/df_{name}_outliers_{percentage}percent.csv')
        # df = pd.read_csv(f'data/numerical/wOutliers/run1/df_{name}_outliers_{percentage}percent.csv')
        data_wOutliers[f'df_{name}_outliers_{percentage}percent'] = df

def perform_clustering_and_scoring(X, linkage_type, distance_matrix):
    clusters = AgglomerativeClustering(n_clusters=NB_CLUSTERS, linkage=linkage_type, metric='precomputed').fit_predict(distance_matrix)
    silhouette_index = silhouette_score(distance_matrix, clusters, metric='precomputed')
    dbi_index = calculate_dbi(distance_matrix, clusters)
    ari_index = adjusted_rand_score(y_encoded, clusters[:len(y_encoded)])
    return clusters, silhouette_index, dbi_index, ari_index

linkage_names = {'single': '\\nearestNeighbourMethod', 'average': '\\averageLinkageMethod', 'complete': '\\farthestNeighbourMethod'}
kmodes_name = '\\kModesMethod'

linkages = ['single', 'average', 'complete']

for name, df in data_wOutliers.items():
    print(f'Processing {name}')
    percentage = name.split('_')[-1].replace('percent', '')
    X, y = split_df(df, number_of_columns=2)
    X = X.astype(str)
    # for column in X.columns:
    #     print(X[column].value_counts())

    distance_matrix_smc = squareform(pdist(X, metric=smc_distance))

    for linkage in linkages:
        clusters, silhouette_index, dbi_index, ari_index = perform_clustering_and_scoring(X, linkage, distance_matrix_smc)
        # print(clusters)
        # print(f'{percentage}\\% & {linkage_names[linkage]} & {silhouette_index:.4f} & {dbi_index:.4f} & {ari_index:.4f} \\\\')


    kmodes = KModes(n_clusters=NB_CLUSTERS)
    clusters_kmodes = kmodes.fit_predict(X)
    silhouette_index_kmodes = silhouette_score(distance_matrix_smc, clusters_kmodes, metric='precomputed')
    dbi_index_kmodes = calculate_dbi(distance_matrix_smc, clusters_kmodes)
    ari_index_kmodes = adjusted_rand_score(y_encoded, clusters_kmodes[:len(y_encoded)])
    # print(f' & {kmodes_name} & {silhouette_index_kmodes:.4f} & {dbi_index_kmodes:.4f} & {ari_index_kmodes:.4f} \\\\')





######################
# Data with outliers #
######################
# local_1percent_nearest = []
# local_5percent_nearest = []
# local_10percent_nearest = []

# global_1percent_nearest = []
# global_5percent_nearest = []
# global_10percent_nearest = []

# contextual_1percent_nearest = []
# contextual_5percent_nearest = []
# contextual_10percent_nearest = []

# collective_1percent_nearest = []
# collective_5percent_nearest = []
# collective_10percent_nearest = []

# local_1percent_average = []
# local_5percent_average = []
# local_10percent_average = []

# global_1percent_average = []
# global_5percent_average = []
# global_10percent_average = []

# contextual_1percent_average = []
# contextual_5percent_average = []
# contextual_10percent_average = []

# collective_1percent_average = []
# collective_5percent_average = []
# collective_10percent_average = []

# local_1percent_farthest = []
# local_5percent_farthest = []
# local_10percent_farthest = []

# global_1percent_farthest = []
# global_5percent_farthest = []
# global_10percent_farthest = []

# contextual_1percent_farthest = []
# contextual_5percent_farthest = []
# contextual_10percent_farthest = []

# collective_1percent_farthest = []
# collective_5percent_farthest = []
# collective_10percent_farthest = []

# local_1percent_kmodes = []
# local_5percent_kmodes = []
# local_10percent_kmodes = []

# global_1percent_kmodes = []
# global_5percent_kmodes = []
# global_10percent_kmodes = []

# contextual_1percent_kmodes = []
# contextual_5percent_kmodes = []
# contextual_10percent_kmodes = []

# collective_1percent_kmodes = []
# collective_5percent_kmodes = []
# collective_10percent_kmodes = []

# outlier_names = ['local', 'global', 'contextual', 'collective']
# outlier_percentages = [1, 5, 10]

# data_wOutliers = {}
# metric_type = 'precomputed'

# for run_number in range(1, 51):
#     for name in outlier_names:
#         for percentage in outlier_percentages:
#             df = pd.read_csv(f'/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/data/categorical/wOutliers/run{run_number}/df_{name}_outliers_{percentage}percent.csv')
#             data_wOutliers[f'df_{name}_outliers_{percentage}percent'] = df

#     for name, df in data_wOutliers.items():
#         # X = df.drop(['Druh', 'IsOutlier'], axis=1)
#         X, y = split_df(df, number_of_columns=2)
#         X = X.astype(str)
#         distance_matrix = squareform(pdist(X, metric=smc_distance))
#         # print(f'Processing {name}')

#         ari_index = adjusted_rand_score(y_encoded, clusters[:len(y_encoded)])

#         nn_model = AgglomerativeClustering(n_clusters=NB_CLUSTERS, metric=metric_type, linkage='single')
#         nn_labels = nn_model.fit_predict(distance_matrix)
#         ave_model = AgglomerativeClustering(n_clusters=NB_CLUSTERS, metric=metric_type, linkage='average')
#         ave_labels = ave_model.fit_predict(distance_matrix)
#         fn_model = AgglomerativeClustering(n_clusters=NB_CLUSTERS, metric=metric_type, linkage='complete')
#         fn_label = fn_model.fit_predict(distance_matrix)
#         kmodes_model = KModes(n_clusters=NB_CLUSTERS)
#         kmodes_labels = kmodes_model.fit_predict(X)

#         si_nn = silhouette_score(distance_matrix, nn_labels, metric=metric_type)
#         si_ave = silhouette_score(distance_matrix, ave_labels, metric=metric_type)
#         si_fn = silhouette_score(distance_matrix, fn_label, metric=metric_type)
#         si_kmodes = silhouette_score(distance_matrix, kmodes_labels, metric=metric_type)

#         db_nn = calculate_dbi(distance_matrix, nn_labels)
#         db_ave = calculate_dbi(distance_matrix, ave_labels)
#         db_fn = calculate_dbi(distance_matrix, fn_label)
#         db_kmodes = calculate_dbi(distance_matrix, kmodes_labels)

#         ari_nn = adjusted_rand_score(y_encoded, nn_labels[:len(y_encoded)])
#         ari_ave = adjusted_rand_score(y_encoded, ave_labels[:len(y_encoded)])
#         ari_fn = adjusted_rand_score(y_encoded, fn_label[:len(y_encoded)])
#         ari_kmodes = adjusted_rand_score(y_encoded, kmodes_labels[:len(y_encoded)])

#         if name == 'df_local_outliers_1percent':
#             local_1percent_nearest.append([si_nn, db_nn, ari_nn])
#             local_1percent_average.append([si_ave, db_ave, ari_ave])
#             local_1percent_farthest.append([si_fn, db_fn, ari_fn])
#             local_1percent_kmodes.append([si_kmodes, db_kmodes, ari_kmodes])
#         elif name == 'df_local_outliers_5percent':
#             local_5percent_nearest.append([si_nn, db_nn, ari_nn])
#             local_5percent_average.append([si_ave, db_ave, ari_ave])
#             local_5percent_farthest.append([si_fn, db_fn, ari_fn])
#             local_5percent_kmodes.append([si_kmodes, db_kmodes, ari_kmodes])
#         elif name == 'df_local_outliers_10percent':
#             local_10percent_nearest.append([si_nn, db_nn, ari_nn])
#             local_10percent_average.append([si_ave, db_ave, ari_ave])
#             local_10percent_farthest.append([si_fn, db_fn, ari_fn])
#             local_10percent_kmodes.append([si_kmodes, db_kmodes, ari_kmodes])
#         elif name == 'df_global_outliers_1percent':
#             global_1percent_nearest.append([si_nn, db_nn, ari_nn])
#             global_1percent_average.append([si_ave, db_ave, ari_ave])
#             global_1percent_farthest.append([si_fn, db_fn, ari_fn])
#             global_1percent_kmodes.append([si_kmodes, db_kmodes, ari_kmodes])
#         elif name == 'df_global_outliers_5percent':
#             global_5percent_nearest.append([si_nn, db_nn, ari_nn])
#             global_5percent_average.append([si_ave, db_ave, ari_ave])
#             global_5percent_farthest.append([si_fn, db_fn, ari_fn])
#             global_5percent_kmodes.append([si_kmodes, db_kmodes, ari_kmodes])
#         elif name == 'df_global_outliers_10percent':
#             global_10percent_nearest.append([si_nn, db_nn, ari_nn])
#             global_10percent_average.append([si_ave, db_ave, ari_ave])
#             global_10percent_farthest.append([si_fn, db_fn, ari_fn])
#             global_10percent_kmodes.append([si_kmodes, db_kmodes, ari_kmodes])
#         elif name == 'df_contextual_outliers_1percent':
#             contextual_1percent_nearest.append([si_nn, db_nn, ari_nn])
#             contextual_1percent_average.append([si_ave, db_ave, ari_ave])
#             contextual_1percent_farthest.append([si_fn, db_fn, ari_fn])
#             contextual_1percent_kmodes.append([si_kmodes, db_kmodes, ari_kmodes])
#         elif name == 'df_contextual_outliers_5percent':
#             contextual_5percent_nearest.append([si_nn, db_nn, ari_nn])
#             contextual_5percent_average.append([si_ave, db_ave, ari_ave])
#             contextual_5percent_farthest.append([si_fn, db_fn, ari_fn])
#             contextual_5percent_kmodes.append([si_kmodes, db_kmodes, ari_kmodes])
#         elif name == 'df_contextual_outliers_10percent':
#             contextual_10percent_nearest.append([si_nn, db_nn, ari_nn])
#             contextual_10percent_average.append([si_ave, db_ave, ari_ave])
#             contextual_10percent_farthest.append([si_fn, db_fn, ari_fn])
#             contextual_10percent_kmodes.append([si_kmodes, db_kmodes, ari_kmodes])
#         elif name == 'df_collective_outliers_1percent':
#             collective_1percent_nearest.append([si_nn, db_nn, ari_nn])
#             collective_1percent_average.append([si_ave, db_ave, ari_ave])
#             collective_1percent_farthest.append([si_fn, db_fn, ari_fn])
#             collective_1percent_kmodes.append([si_kmodes, db_kmodes, ari_kmodes])
#         elif name == 'df_collective_outliers_5percent':
#             collective_5percent_nearest.append([si_nn, db_nn, ari_nn])
#             collective_5percent_average.append([si_ave, db_ave, ari_ave])
#             collective_5percent_farthest.append([si_fn, db_fn, ari_fn])
#             collective_5percent_kmodes.append([si_kmodes, db_kmodes, ari_kmodes])
#         elif name == 'df_collective_outliers_10percent':
#             collective_10percent_nearest.append([si_nn, db_nn, ari_nn])
#             collective_10percent_average.append([si_ave, db_ave, ari_ave])
#             collective_10percent_farthest.append([si_fn, db_fn, ari_fn])
#             collective_10percent_kmodes.append([si_kmodes, db_kmodes, ari_kmodes])

        

# local_1_nn = np.mean(np.array(local_1percent_nearest), axis=0)
# local_1_ave = np.mean(np.array(local_1percent_average), axis=0)
# local_1_fn = np.mean(np.array(local_1percent_farthest), axis=0)
# local_1_kmodes = np.mean(np.array(local_1percent_kmodes), axis=0)

# local_5_nn = np.mean(np.array(local_5percent_nearest), axis=0)
# local_5_ave = np.mean(np.array(local_5percent_average), axis=0)
# local_5_fn = np.mean(np.array(local_5percent_farthest), axis=0)
# local_5_kmodes = np.mean(np.array(local_5percent_kmodes), axis=0)

# local_10_nn = np.mean(np.array(local_10percent_nearest), axis=0)
# local_10_ave = np.mean(np.array(local_10percent_average), axis=0)
# local_10_fn = np.mean(np.array(local_10percent_farthest), axis=0)
# local_10_kmodes = np.mean(np.array(local_10percent_kmodes), axis=0)

# global_1_nn = np.mean(np.array(global_1percent_nearest), axis=0)
# global_1_ave = np.mean(np.array(global_1percent_average), axis=0)
# global_1_fn = np.mean(np.array(global_1percent_farthest), axis=0)
# global_1_kmodes = np.mean(np.array(global_1percent_kmodes), axis=0)

# global_5_nn = np.mean(np.array(global_5percent_nearest), axis=0)
# global_5_ave = np.mean(np.array(global_5percent_average), axis=0)
# global_5_fn = np.mean(np.array(global_5percent_farthest), axis=0)
# global_5_kmodes = np.mean(np.array(global_5percent_kmodes), axis=0)

# global_10_nn = np.mean(np.array(global_10percent_nearest), axis=0)
# global_10_ave = np.mean(np.array(global_10percent_average), axis=0)
# global_10_fn = np.mean(np.array(global_10percent_farthest), axis=0)
# global_10_kmodes = np.mean(np.array(global_10percent_kmodes), axis=0)

# contextual_1_nn = np.mean(np.array(contextual_1percent_nearest), axis=0)
# contextual_1_ave = np.mean(np.array(contextual_1percent_average), axis=0)
# contextual_1_fn = np.mean(np.array(contextual_1percent_farthest), axis=0)
# contextual_1_kmodes = np.mean(np.array(contextual_1percent_kmodes), axis=0)

# contextual_5_nn = np.mean(np.array(contextual_5percent_nearest), axis=0)
# contextual_5_ave = np.mean(np.array(contextual_5percent_average), axis=0)
# contextual_5_fn = np.mean(np.array(contextual_5percent_farthest), axis=0)
# contextual_5_kmodes = np.mean(np.array(contextual_5percent_kmodes), axis=0)

# contextual_10_nn = np.mean(np.array(contextual_10percent_nearest), axis=0)
# contextual_10_ave = np.mean(np.array(contextual_10percent_average), axis=0)
# contextual_10_fn = np.mean(np.array(contextual_10percent_farthest), axis=0)
# contextual_10_kmodes = np.mean(np.array(contextual_10percent_kmodes), axis=0)

# collective_1_nn = np.mean(np.array(collective_1percent_nearest), axis=0)
# collective_1_ave = np.mean(np.array(collective_1percent_average), axis=0)
# collective_1_fn = np.mean(np.array(collective_1percent_farthest), axis=0)
# collective_1_kmodes = np.mean(np.array(collective_1percent_kmodes), axis=0)

# collective_5_nn = np.mean(np.array(collective_5percent_nearest), axis=0)
# collective_5_ave = np.mean(np.array(collective_5percent_average), axis=0)
# collective_5_fn = np.mean(np.array(collective_5percent_farthest), axis=0)
# collective_5_kmodes = np.mean(np.array(collective_5percent_kmodes), axis=0)

# collective_10_nn = np.mean(np.array(collective_10percent_nearest), axis=0)
# collective_10_ave = np.mean(np.array(collective_10percent_average), axis=0)
# collective_10_fn = np.mean(np.array(collective_10percent_farthest), axis=0)
# collective_10_kmodes = np.mean(np.array(collective_10percent_kmodes), axis=0)

# print('Local 1%')
# print(f"& {local_1_nn[0]:.4f} & {local_1_nn[1]:.4f} & {local_1_nn[2]:.4f} \\\\")
# print(f"& {local_1_ave[0]:.4f} & {local_1_ave[1]:.4f} & {local_1_ave[2]:.4f} \\\\")
# print(f"& {local_1_fn[0]:.4f} & {local_1_fn[1]:.4f} & {local_1_fn[2]:.4f} \\\\")
# print(f"& {local_1_kmodes[0]:.4f} & {local_1_kmodes[1]:.4f} & {local_1_kmodes[2]:.4f} \\\\")

# print('Local 5%')
# print(f"& {local_5_nn[0]:.4f} & {local_5_nn[1]:.4f} & {local_5_nn[2]:.4f} \\\\")
# print(f"& {local_5_ave[0]:.4f} & {local_5_ave[1]:.4f} & {local_5_ave[2]:.4f} \\\\")
# print(f"& {local_5_fn[0]:.4f} & {local_5_fn[1]:.4f} & {local_5_fn[2]:.4f} \\\\")
# print(f"& {local_5_kmodes[0]:.4f} & {local_5_kmodes[1]:.4f} & {local_5_kmodes[2]:.4f} \\\\")

# print('Local 10%')
# print(f"& {local_10_nn[0]:.4f} & {local_10_nn[1]:.4f} & {local_10_nn[2]:.4f} \\\\")
# print(f"& {local_10_ave[0]:.4f} & {local_10_ave[1]:.4f} & {local_10_ave[2]:.4f} \\\\")
# print(f"& {local_10_fn[0]:.4f} & {local_10_fn[1]:.4f} & {local_10_fn[2]:.4f} \\\\")
# print(f"& {local_10_kmodes[0]:.4f} & {local_10_kmodes[1]:.4f} & {local_10_kmodes[2]:.4f} \\\\")

# print('Global 1%')
# print(f"& {global_1_nn[0]:.4f} & {global_1_nn[1]:.4f} & {global_1_nn[2]:.4f} \\\\")
# print(f"& {global_1_ave[0]:.4f} & {global_1_ave[1]:.4f} & {global_1_ave[2]:.4f} \\\\")
# print(f"& {global_1_fn[0]:.4f} & {global_1_fn[1]:.4f} & {global_1_fn[2]:.4f} \\\\")
# print(f"& {global_1_kmodes[0]:.4f} & {global_1_kmodes[1]:.4f} & {global_1_kmodes[2]:.4f} \\\\")

# print('Global 5%')
# print(f"& {global_5_nn[0]:.4f} & {global_5_nn[1]:.4f} & {global_5_nn[2]:.4f} \\\\")
# print(f"& {global_5_ave[0]:.4f} & {global_5_ave[1]:.4f} & {global_5_ave[2]:.4f} \\\\")
# print(f"& {global_5_fn[0]:.4f} & {global_5_fn[1]:.4f} & {global_5_fn[2]:.4f} \\\\")
# print(f"& {global_5_kmodes[0]:.4f} & {global_5_kmodes[1]:.4f} & {global_5_kmodes[2]:.4f} \\\\")

# print('Global 10%')
# print(f"& {global_10_nn[0]:.4f} & {global_10_nn[1]:.4f} & {global_10_nn[2]:.4f} \\\\")
# print(f"& {global_10_ave[0]:.4f} & {global_10_ave[1]:.4f} & {global_10_ave[2]:.4f} \\\\")
# print(f"& {global_10_fn[0]:.4f} & {global_10_fn[1]:.4f} & {global_10_fn[2]:.4f} \\\\")
# print(f"& {global_10_kmodes[0]:.4f} & {global_10_kmodes[1]:.4f} & {global_10_kmodes[2]:.4f} \\\\")

# print('Contextual 1%')
# print(f"& {contextual_1_nn[0]:.4f} & {contextual_1_nn[1]:.4f} & {contextual_1_nn[2]:.4f} \\\\")
# print(f"& {contextual_1_ave[0]:.4f} & {contextual_1_ave[1]:.4f} & {contextual_1_ave[2]:.4f} \\\\")
# print(f"& {contextual_1_fn[0]:.4f} & {contextual_1_fn[1]:.4f} & {contextual_1_fn[2]:.4f} \\\\")
# print(f"& {contextual_1_kmodes[0]:.4f} & {contextual_1_kmodes[1]:.4f} & {contextual_1_kmodes[2]:.4f} \\\\")
# print('Contextual 5%')
# print(f"& {contextual_5_nn[0]:.4f} & {contextual_5_nn[1]:.4f} & {contextual_5_nn[2]:.4f} \\\\")
# print(f"& {contextual_5_ave[0]:.4f} & {contextual_5_ave[1]:.4f} & {contextual_5_ave[2]:.4f} \\\\")
# print(f"& {contextual_5_fn[0]:.4f} & {contextual_5_fn[1]:.4f} & {contextual_5_fn[2]:.4f} \\\\")
# print(f"& {contextual_5_kmodes[0]:.4f} & {contextual_5_kmodes[1]:.4f} & {contextual_5_kmodes[2]:.4f} \\\\")
# print('Contextual 10%')
# print(f"& {contextual_10_nn[0]:.4f} & {contextual_10_nn[1]:.4f} & {contextual_10_nn[2]:.4f} \\\\")
# print(f"& {contextual_10_ave[0]:.4f} & {contextual_10_ave[1]:.4f} & {contextual_10_ave[2]:.4f} \\\\")
# print(f"& {contextual_10_fn[0]:.4f} & {contextual_10_fn[1]:.4f} & {contextual_10_fn[2]:.4f} \\\\")
# print(f"& {contextual_10_kmodes[0]:.4f} & {contextual_10_kmodes[1]:.4f} & {contextual_10_kmodes[2]:.4f} \\\\")
# print('Collective 1%')
# print(f"& {collective_1_nn[0]:.4f} & {collective_1_nn[1]:.4f} & {collective_1_nn[2]:.4f} \\\\")
# print(f"& {collective_1_ave[0]:.4f} & {collective_1_ave[1]:.4f} & {collective_1_ave[2]:.4f} \\\\")
# print(f"& {collective_1_fn[0]:.4f} & {collective_1_fn[1]:.4f} & {collective_1_fn[2]:.4f} \\\\")
# print(f"& {collective_1_kmodes[0]:.4f} & {collective_1_kmodes[1]:.4f} & {collective_1_kmodes[2]:.4f} \\\\")
# print('Collective 5%')
# print(f"& {collective_5_nn[0]:.4f} & {collective_5_nn[1]:.4f} & {collective_5_nn[2]:.4f} \\\\")
# print(f"& {collective_5_ave[0]:.4f} & {collective_5_ave[1]:.4f} & {collective_5_ave[2]:.4f} \\\\")
# print(f"& {collective_5_fn[0]:.4f} & {collective_5_fn[1]:.4f} & {collective_5_fn[2]:.4f} \\\\")
# print(f"& {collective_5_kmodes[0]:.4f} & {collective_5_kmodes[1]:.4f} & {collective_5_kmodes[2]:.4f} \\\\")
# print('Collective 10%')
# print(f"& {collective_10_nn[0]:.4f} & {collective_10_nn[1]:.4f} & {collective_10_nn[2]:.4f} \\\\")
# print(f"& {collective_10_ave[0]:.4f} & {collective_10_ave[1]:.4f} & {collective_10_ave[2]:.4f} \\\\")
# print(f"& {collective_10_fn[0]:.4f} & {collective_10_fn[1]:.4f} & {collective_10_fn[2]:.4f} \\\\")
# print(f"& {collective_10_kmodes[0]:.4f} & {collective_10_kmodes[1]:.4f} & {collective_10_kmodes[2]:.4f} \\\\")








# outlier_names = ['local', 'global', 'contextual', 'collective']
# outlier_percentages = [1, 5, 10]
# clustering_methods = ['nearest', 'average', 'farthest', 'kmodes']

# # Initialize results storage
# results = {name: {f"{percent}%": {method: [] for method in clustering_methods} 
#                   for percent in outlier_percentages} 
#            for name in outlier_names}

# def calculate_metrics(X, y):
#     """Calculate silhouette, DBI, and ARI for clustering methods."""
#     distance_matrix = squareform(pdist(X, metric=smc_distance))
#     metrics = {}
#     methods = {
#         'nearest': ('single', AgglomerativeClustering(n_clusters=NB_CLUSTERS, metric='precomputed', linkage='single')),
#         'average': ('average', AgglomerativeClustering(n_clusters=NB_CLUSTERS, metric='precomputed', linkage='average')),
#         'farthest': ('complete', AgglomerativeClustering(n_clusters=NB_CLUSTERS, metric='precomputed', linkage='complete')),
#         'kmodes': ('kmodes', KModes(n_clusters=NB_CLUSTERS))
#     }

#     for method, (linkage, model) in methods.items():
#         if method == 'kmodes':
#             labels = model.fit_predict(X)
#         else:
#             labels = model.fit_predict(distance_matrix)
        
#         si = silhouette_score(distance_matrix, labels, metric='precomputed')
#         ari = adjusted_rand_score(y_encoded, labels[:len(y_encoded)])
#         dbi = calculate_dbi(distance_matrix, labels)
        
#         # Ensure the metrics are returned as a list
#         metrics[method] = [si, dbi, ari]

#     # Ensure a dictionary with the correct structure is returned
#     return metrics

# for run_number in range(1, 51):
#     for outlier_type in outlier_names:
#         for percentage in outlier_percentages:
#             # df = pd.read_csv(f'/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/data/categorical/wOutliers/run{run_number}/df_{name}_outliers_{percentage}percent.csv')            
#             df_path = f'/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/data/categorical/wOutliers/run{run_number}/df_{outlier_type}_outliers_{percentage}percent.csv'
#             df = pd.read_csv(df_path)
#             X, y = split_df(df)
#             X = X.astype(str)  
            
#             metric_results = calculate_metrics(X, y)
#             for method, values in metric_results.items():
#                 results[outlier_type][f"{percentage}%"][method].append(values)

# for outlier_type, type_data in results.items():
#     for percentage, method_data in type_data.items():
#         for method, values in method_data.items():
#             mean_values = np.mean(values, axis=0)
#             print(f"{outlier_type} {percentage} {method}: Silhouette={mean_values[0]:.4f}, DBI={mean_values[1]:.4f}, ARI={mean_values[2]:.4f}")



##############################
# Data with outliers removed #
##############################
outlier_detection_methods = ['Kmodes', 'CBRW', 'FPOF']
data_wOutliers_removed_KModes = {}
data_wOutliers_removed_CBRW = {}
data_wOutliers_removed_FPOF = {}

for name in outlier_names:
    for percentage in outlier_percentages:
        for method_name in outlier_detection_methods:
            df = pd.read_csv(f'data/categorical/wOutliers/run1/removed/{method_name}/df_{name}_outliers_{percentage}percent_removed_{method_name}.csv')
            if method_name == 'Kmodes':
                data_wOutliers_removed_KModes[f'df_{name}_outliers_{percentage}percent_removed_{method_name}'] = df
            elif method_name == 'CBRW':
                data_wOutliers_removed_CBRW[f'df_{name}_outliers_{percentage}percent_removed_{method_name}'] = df
            elif method_name == 'FPOF':
                data_wOutliers_removed_FPOF[f'df_{name}_outliers_{percentage}percent_removed_{method_name}'] = df

# print(data_wOutliers_removed_KModes.keys())
# print(data_wOutliers_removed_CBRW.keys())
# print(data_wOutliers_removed_FPOF.keys())
        
for name, df in data_wOutliers_removed_KModes.items():
    print(f'\n\nProcessing {name}')
    percentage = name.split('_')[-3].replace('percent', '')
    X, y = split_df(df, number_of_columns=2)
    X = X.astype(str)
    distance_matrix_smc = squareform(pdist(X, metric=smc_distance))

    for linkage in linkages:
        clusters, silhouette_index, dbi_index, ari_index = perform_clustering_and_scoring(X, linkage, distance_matrix_smc)
        # print(f'{percentage}\\% & {linkage_names[linkage]} & {silhouette_index:.4f} & {dbi_index:.4f} & {ari_index:.4f} \\\\')

    kmodes = KModes(n_clusters=NB_CLUSTERS)
    clusters_kmodes = kmodes.fit_predict(X)
    silhouette_index_kmodes = silhouette_score(distance_matrix_smc, clusters_kmodes, metric='precomputed')
    dbi_index_kmodes = calculate_dbi(distance_matrix_smc, clusters_kmodes)
    ari_index_kmodes = adjusted_rand_score(y_encoded, clusters_kmodes[:len(y_encoded)])
    # print(f' & {kmodes_name} & {silhouette_index_kmodes:.4f} & {dbi_index_kmodes:.4f} & {ari_index_kmodes:.4f}  \\\\')

for name, df in data_wOutliers_removed_CBRW.items():
    print(f'\n\nProcessing {name}')
    percentage = name.split('_')[-3].replace('percent', '')
    X, y = split_df(df, number_of_columns=2)
    X = X.astype(str)
    distance_matrix_smc = squareform(pdist(X, metric=smc_distance))

    for linkage in linkages:
        clusters, silhouette_index, dbi_index, ari_index = perform_clustering_and_scoring(X, linkage, distance_matrix_smc)
        # print(f'{percentage}\\% & {linkage_names[linkage]} & {silhouette_index:.4f} & {dbi_index:.4f} & {ari_index:.4f} \\\\')

    kmodes = KModes(n_clusters=NB_CLUSTERS)
    clusters_kmodes = kmodes.fit_predict(X)
    silhouette_index_kmodes = silhouette_score(distance_matrix_smc, clusters_kmodes, metric='precomputed')
    dbi_index_kmodes = calculate_dbi(distance_matrix_smc, clusters_kmodes)
    ari_index_kmodes = adjusted_rand_score(y_encoded, clusters_kmodes[:len(y_encoded)])
    # print(f' & {kmodes_name} & {silhouette_index_kmodes:.4f} & {dbi_index_kmodes:.4f} & {ari_index_kmodes:.4f}  \\\\')

for name, df in data_wOutliers_removed_FPOF.items():
    print(f'\n\nProcessing {name}')
    percentage = name.split('_')[-3].replace('percent', '')
    X, y = split_df(df, number_of_columns=2)
    X = X.astype(str)
    distance_matrix_smc = squareform(pdist(X, metric=smc_distance))
    # print(X.head())

    for linkage in linkages:
        clusters, silhouette_index, dbi_index, ari_index = perform_clustering_and_scoring(X, linkage, distance_matrix_smc)
        # print(f'{percentage}\\% & {linkage_names[linkage]} & {silhouette_index:.4f} & {dbi_index:.4f}  & {ari_index:.4f} \\\\')

    kmodes = KModes(n_clusters=NB_CLUSTERS)
    clusters_kmodes = kmodes.fit_predict(X)
    silhouette_index_kmodes = silhouette_score(distance_matrix_smc, clusters_kmodes, metric='precomputed')
    dbi_index_kmodes = calculate_dbi(distance_matrix_smc, clusters_kmodes)
    ari_index_kmodes = adjusted_rand_score(y_encoded, clusters_kmodes[:len(y_encoded)])
    # print(f' & {kmodes_name} & {silhouette_index_kmodes:.4f} & {dbi_index_kmodes:.4f} & {ari_index_kmodes:.4f}  \\\\')








###################
# with Simulation #
###################

outlier_names = ['local', 'global', 'contextual', 'collective']
outlier_percentages = [1, 5, 10]
outlier_detection_methods = ['KModes', 'CBRW', 'FPOF']
runs = range(1, 51)
n_cluster = 3
linkages = ['single', 'average', 'complete']

data_wOutliers_removed = {
    method: {f'{name}_{percentage}': [] for name in outlier_names for percentage in outlier_percentages}
    for method in outlier_detection_methods
}

for run in runs:
    for name in outlier_names:
        for percentage in outlier_percentages:
            for method_name in outlier_detection_methods:
                filepath = f'/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/data/categorical/wOutliers/run{run}/removed/{method_name}/df_{name}_outliers_{percentage}percent_removed_{method_name}.csv'
                df = pd.read_csv(filepath)
                data_wOutliers_removed[method_name][f'{name}_{percentage}'].append(df)

results = {
    outlier_type: {
        percentage: {
            method: {
                'Agglomerative': {linkage: {'silhouette_scores': [], 'db_scores': []} for linkage in linkages},
                'KModes': {'silhouette_scores': [], 'db_scores': []}
            } for method in outlier_detection_methods
        }
        for percentage in outlier_percentages
    }
    for outlier_type in outlier_names
}

for method_name, datasets in data_wOutliers_removed.items():
    for key, dfs in datasets.items():
        outlier_type, str_percentage = key.split('_')
        percentage = int(str_percentage)
        for df in dfs:
            X = df.drop(['Druh', 'IsOutlier'], axis=1)
            X = X.astype(str)
            distance_matrix = squareform(pdist(X, metric=smc_distance))

            for linkage in linkages:
                model_agg = AgglomerativeClustering(n_clusters=n_cluster, metric='precomputed', linkage=linkage)
                labels_agg = model_agg.fit_predict(distance_matrix)
                silhouette = silhouette_score(distance_matrix, labels_agg, metric = 'precomputed')
                db_index = calculate_dbi(distance_matrix, labels_agg)

                results[outlier_type][percentage][method_name]['Agglomerative'][linkage]['silhouette_scores'].append(silhouette)
                results[outlier_type][percentage][method_name]['Agglomerative'][linkage]['db_scores'].append(db_index)
                print(f'appended {outlier_type} {percentage} {method_name} {linkage} values {silhouette} {db_index}')
            
            model_kmodes = KModes(n_clusters=NB_CLUSTERS)
            labels_kmodes = model_kmodes.fit_predict(X)
            silhouette_kmodes = silhouette_score(distance_matrix, labels_kmodes, metric = 'precomputed')
            db_index_kmodes = calculate_dbi(distance_matrix, labels_kmodes)

            results[outlier_type][percentage][method_name]['KModes']['silhouette_scores'].append(silhouette_kmodes)
            results[outlier_type][percentage][method_name]['KModes']['db_scores'].append(db_index_kmodes)
            print(f'appended {outlier_type} {percentage} {method_name} values {silhouette_kmodes} {db_index_kmodes}')

for outlier_type, types in results.items():
    print(f"\nResults for Outlier Type: {outlier_type}")
    for percentage, percent_data in types.items():
        print(f"  Percentage: {percentage}%")
        for method, method_data in percent_data.items():
            print(f"  Method: {method}")
            for clustering_method, data in method_data.items():
                if clustering_method == 'KModes':
                    avg_silhouette = np.mean(data['silhouette_scores'])
                    avg_db = np.mean(data['db_scores'])
                    # print(f"    {method} - KMeans: Avg Silhouette: {avg_silhouette:.4f} & {avg_db:.4f} \\\\")
                    print(f" &   & \\kModesMethod & {avg_silhouette:.4f} & {avg_db:.4f} \\\\")
                else:
                    for linkage, scores in data.items():
                        # if method == 'IsolationForest' and outlier_type == 'contextual':
                            # print(f"{percentage}%, {clustering_method}: {scores['silhouette_scores']}")
                        avg_silhouette = np.mean(scores['silhouette_scores'])
                        avg_db = np.mean(scores['db_scores'])
                        # print(f"    {method} - Agglomerative ({linkage}): Avg Silhouette: {avg_silhouette:.4f} & {avg_db:.4f} \\\\")
                        if linkage == 'single':
                            print(f" & & \\nearestNeighbourMethod &  {avg_silhouette:.4f} & {avg_db:.4f} \\\\")
                        elif linkage == 'average':
                            print(f" & & \\averageLinkageMethod &  {avg_silhouette:.4f} & {avg_db:.4f} \\\\")
                        elif linkage == 'complete':
                            print(f" & & \\farthestNeighbourMethod &  {avg_silhouette:.4f} & {avg_db:.4f} \\\\")

with open('/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/clustering_results_categorical.json', 'w') as f:
    json.dump(results, f, indent=4)


####################################################################
# Plot difference between original and dataset with added outleirs #
####################################################################

# local outliers
local_outliers_ARI = [
    {
        'single': 0.1142,
        'average': 0.5565,
        'complete': 0.0800,
        'kmodes': 0.3876
    },
    {
        'single': 0.0113,
        'average': 0.5915,
        'complete': 0.1646,
        'kmodes': 0.4680
    },
    {
        'single': 0.0000,
        'average': 0.5821,
        'complete': 0.1322,
        'kmodes': 0.4898
    }
]

# global outliers
global_outliers_ARI = [
    {
        'single': 0.2051,
        'average': 0.5665,
        'complete': 0.0775,
        'kmodes': 0.4311
    },
    {
        'single': 0.0227,
        'average': 0.5704,
        'complete': 0.1299,
        'kmodes': 0.4969
    },
    {
        'single': 0.0000,
        'average': 0.5521,
        'complete': 0.2163,
        'kmodes': 0.5491
    }
]

# contextual outliers
contextual_outliers_ARI = [
    {
        'single': 0.0340,
        'average': 0.5701,
        'complete': 0.0962,
        'kmodes': 0.3933
    },
    {
        'single': 0.0000,
        'average': 0.5755,
        'complete': 0.0796,
        'kmodes': 0.3915
    },
    {
        'single': 0.0000,
        'average': 0.5759,
        'complete': 0.0747,
        'kmodes': 0.3822
    }
]

# collective outliers
collective_outliers_ARI = [
    {
        'single': 0.0000,
        'average': 0.5552,
        'complete': 0.1169,
        'kmodes': 0.4143
    },
    {
        'single': 0.0000,
        'average': 0.5418,
        'complete': 0.3513,
        'kmodes': 0.4143
    },
    {
        'single': 0.0001,
        'average': 0.5561,
        'complete': 0.2490,
        'kmodes': 0.4143
    }
]

initial_ARI_values = {
    'single': 0.0002,
    'average': 0.5600,
    'complete': 0.1475,
    'kmodes': 0.4143
}

name_mapping = {
    'single': 'nejbližšího souseda',
    'average': 'průměrné vazby',
    'complete': 'nejvzdálenějšího souseda',
    'kmodes': 'k-módů'
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

    # plt.savefig(f'plots/{outlier_type}_ari_diff_categorical.pdf', format='pdf')
    # plt.show()



original_scores_local = {
    '1%': {
        'nejbližšího souseda': 0.0559,
        'průměrné vazby': 0.3431,
        'nejvzdálenějšího souseda': 0.0980,
        'k-módů': 0.2742
    },
    '5%': {
        'nejbližšího souseda': 0.0081,
        'průměrné vazby': 0.3538,
        'nejvzdálenějšího souseda': 0.1107,
        'k-módů': 0.3012
    },
    '10%': {
        'nejbližšího souseda': 0.0075,
        'průměrné vazby': 0.3498,
        'nejvzdálenějšího souseda': 0.1058,
        'k-módů': 0.3099
    }
}

original_scores_global = {
    '1%': {
        'nejbližšího souseda': 0.1641,
        'průměrné vazby': 0.3558,
        'nejvzdálenějšího souseda': 0.1009,
        'k-módů': 0.2850
    },
    '5%': {
        'nejbližšího souseda': 0.2026,
        'průměrné vazby': 0.3935,
        'nejvzdálenějšího souseda': 0.2267,
        'k-módů': 0.3379
    },
    '10%': {
        'nejbližšího souseda': 0.1906,
        'průměrné vazby': 0.3930,
        'nejvzdálenějšího souseda': 0.2666,
        'k-módů': 0.3762
    }
}

original_scores_contextual = {
    '1%': {
        'nejbližšího souseda': -0.0032,
        'průměrné vazby': 0.3344,
        'nejvzdálenějšího souseda': 0.1221,
        'k-módů': 0.2715
    },
    '5%': {
        'nejbližšího souseda': -0.0071,
        'průměrné vazby': 0.3070,
        'nejvzdálenějšího souseda': 0.0692,
        'k-módů': 0.2431
    },
    '10%': {
        'nejbližšího souseda': -0.0270,
        'průměrné vazby': 0.2821,
        'nejvzdálenějšího souseda': 0.0667,
        'k-módů': 0.2219
    }
}
original_scores_collective = {
    '1%': {
        'nejbližšího souseda': -0.0197,
        'průměrné vazby': 0.3307,
        'nejvzdálenějšího souseda': 0.1063,
        'k-módů': 0.2900
    },
    '5%': {
        'nejbližšího souseda': 0.0234,
        'průměrné vazby': 0.3188,
        'nejvzdálenějšího souseda': 0.2182,
        'k-módů': 0.2809
    },
    '10%': {
        'nejbližšího souseda': -0.0294,
        'průměrné vazby': 0.3010,
        'nejvzdálenějšího souseda': 0.1766,
        'k-módů': 0.2733
    }
}

removed_scores_local_kmodes = {
    '1%': {
        'nejbližšího souseda': 0.0636,
        'průměrné vazby': 0.3434,
        'nejvzdálenějšího souseda': 0.1029,
        'k-módů': 0.2773
    },
    '5%': {
        'nejbližšího souseda': 0.0175,
        'průměrné vazby': 0.3630,
        'nejvzdálenějšího souseda': 0.1055,
        'k-módů': 0.3250
    },
    '10%': {
        'nejbližšího souseda': 0.0300,
        'průměrné vazby': 0.3783,
        'nejvzdálenějšího souseda': 0.1355,
        'k-módů': 0.3552
    }
}
removed_scores_global_kmodes = {
    '1%': {
        'nejbližšího souseda': 0.1664,
        'průměrné vazby': 0.3631,
        'nejvzdálenějšího souseda': 0.0945,
        'k-módů': 0.2923
    },
    '5%': {
        'nejbližšího souseda': 0.2002,
        'průměrné vazby': 0.4057,
        'nejvzdálenějšího souseda': 0.2425,
        'k-módů': 0.3733
    },
    '10%': {
        'nejbližšího souseda': 0.2097,
        'průměrné vazby': 0.4487,
        'nejvzdálenějšího souseda': 0.3577,
        'k-módů': 0.4574
    }
}

removed_scores_contextual_kmodes = {
    '1%': {
        'nejbližšího souseda': 0.0023,
        'průměrné vazby': 0.3352,
        'nejvzdálenějšího souseda': 0.1229,
        'k-módů': 0.2717
    },
    '5%': {
        'nejbližšího souseda': -0.0145,
        'průměrné vazby': 0.3254,
        'nejvzdálenějšího souseda': 0.0817,
        'k-módů': 0.2645
    },
    '10%': {
        'nejbližšího souseda': -0.0257,
        'průměrné vazby': 0.3129,
        'nejvzdálenějšího souseda': 0.0950,
        'k-módů': 0.2817
    }
}

removed_scores_collective_kmodes = {
    '1%': {
        'nejbližšího souseda': 0.0035,
        'průměrné vazby': 0.3323,
        'nejvzdálenějšího souseda': 0.1081,
        'k-módů': 0.2865
    },
    '5%': {
        'nejbližšího souseda': 0.0397,
        'průměrné vazby': 0.3357,
        'nejvzdálenějšího souseda': 0.2280,
        'k-módů': 0.3001
    },
    '10%': {
        'nejbližšího souseda': 0.0064,
        'průměrné vazby': 0.3353,
        'nejvzdálenějšího souseda': 0.2406,
        'k-módů': 0.3242
    }
}

removed_scores_local_cbrw = {
    '1%': {
        'nejbližšího souseda': 0.0830,
        'průměrné vazby': 0.3466,
        'nejvzdálenějšího souseda': 0.1008,
        'k-módů': 0.2785
    },
    '5%': {
        'nejbližšího souseda': 0.0340,
        'průměrné vazby': 0.3789,
        'nejvzdálenějšího souseda': 0.1111,
        'k-módů': 0.3161
    },
    '10%': {
        'nejbližšího souseda': 0.0093,
        'průměrné vazby': 0.3834,
        'nejvzdálenějšího souseda': 0.1325,
        'k-módů': 0.3466
    }
}
removed_scores_global_cbrw = {
    '1%': {
        'nejbližšího souseda': 0.1613,
        'průměrné vazby': 0.3669,
        'nejvzdálenějšího souseda': 0.1387,
        'k-módů': 0.2889
    },
    '5%': {
        'nejbližšího souseda': 0.2341,
        'průměrné vazby': 0.4226,
        'nejvzdálenějšího souseda': 0.2063,
        'k-módů': 0.3684
    },
    '10%': {
        'nejbližšího souseda': 0.2714,
        'průměrné vazby': 0.4570,
        'nejvzdálenějšího souseda': 0.3244,
        'k-módů': 0.4469
    }
}

removed_scores_contextual_cbrw = {
    '1%': {
        'nejbližšího souseda': 0.0004,
        'průměrné vazby': 0.3373,
        'nejvzdálenějšího souseda': 0.1289,
        'k-módů': 0.2752
    },
    '5%': {
        'nejbližšího souseda': -0.0271,
        'průměrné vazby': 0.3300,
        'nejvzdálenějšího souseda': 0.0687,
        'k-módů': 0.2653
    },
    '10%': {
        'nejbližšího souseda': -0.0374,
        'průměrné vazby': 0.3322,
        'nejvzdálenějšího souseda': 0.0812,
        'k-módů': 0.2634
    }
}

removed_scores_collective_cbrw = {
    '1%': {
        'nejbližšího souseda': -0.0323,
        'průměrné vazby': 0.3365,
        'nejvzdálenějšího souseda': 0.1253,
        'k-módů': 0.2941
    },
    '5%': {
        'nejbližšího souseda': 0.0096,
        'průměrné vazby': 0.3409,
        'nejvzdálenějšího souseda': 0.2413,
        'k-módů': 0.3047
    },
    '10%': {
        'nejbližšího souseda': -0.0075,
        'průměrné vazby': 0.3597,
        'nejvzdálenějšího souseda': 0.2606,
        'k-módů': 0.3059
    }
}

removed_scores_local_FPOF = {
    '1%': {
        'nejbližšího souseda': 0.0624,
        'průměrné vazby': 0.3445,
        'nejvzdálenějšího souseda': 0.1025,
        'k-módů': 0.2790
    },
    '5%': {
        'nejbližšího souseda': 0.0116,
        'průměrné vazby': 0.3807,
        'nejvzdálenějšího souseda': 0.1338,
        'k-módů': 0.3206
    },
    '10%': {
        'nejbližšího souseda': 0.0050,
        'průměrné vazby': 0.4082,
        'nejvzdálenějšího souseda': 0.1482,
        'k-módů': 0.3509
    }
}
removed_scores_global_FPOF = {
    '1%': {
        'nejbližšího souseda': 0.1389,
        'průměrné vazby': 0.3662,
        'nejvzdálenějšího souseda': 0.1164,
        'k-módů': 0.2897
    },
    '5%': {
        'nejbližšího souseda': 0.2333,
        'průměrné vazby': 0.4244,
        'nejvzdálenějšího souseda': 0.2268,
        'k-módů': 0.3717
    },
    '10%': {
        'nejbližšího souseda': 0.2500,
        'průměrné vazby': 0.4795,
        'nejvzdálenějšího souseda': 0.3340,
        'k-módů': 0.4570
    }
}

removed_scores_contextual_FPOF = {
    '1%': {
        'nejbližšího souseda': -0.0017,
        'průměrné vazby': 0.3366,
        'nejvzdálenějšího souseda': 0.1272,
        'k-módů': 0.2754
    },
    '5%': {
        'nejbližšího souseda': -0.0337,
        'průměrné vazby': 0.3371,
        'nejvzdálenějšího souseda': 0.0773,
        'k-módů': 0.2585
    },
    '10%': {
        'nejbližšího souseda': -0.0368,
        'průměrné vazby': 0.3354,
        'nejvzdálenějšího souseda': 0.0752,
        'k-módů': 0.2557
    }
}

removed_scores_collective_FPOF = {
    '1%': {
        'nejbližšího souseda': -0.0323,
        'průměrné vazby': 0.3365,
        'nejvzdálenějšího souseda': 0.1253,
        'k-módů': 0.2940
    },
    '5%': {
        'nejbližšího souseda': 0.0180,
        'průměrné vazby': 0.3466,
        'nejvzdálenějšího souseda': 0.2487,
        'k-módů': 0.2923
    },
    '10%': {
        'nejbližšího souseda': 0.0027,
        'průměrné vazby': 0.3587,
        'nejvzdálenějšího souseda': 0.2542,
        'k-módů': 0.2822
    }
}

removed_methods_local = {
    'k-módů': removed_scores_local_kmodes,
    'CBRW': removed_scores_local_cbrw,
    'FPOF': removed_scores_local_FPOF
}

removed_methods_global = {
    'k-módů': removed_scores_global_kmodes,
    'CBRW': removed_scores_global_cbrw,
    'FPOF': removed_scores_global_FPOF
}

removed_methods_contextual = {
    'k-módů': removed_scores_contextual_kmodes,
    'CBRW': removed_scores_contextual_cbrw,
    'FPOF': removed_scores_contextual_FPOF
}

removed_methods_collective = {
    'k-módů': removed_scores_collective_kmodes,
    'CBRW': removed_scores_collective_cbrw,
    'FPOF': removed_scores_collective_FPOF
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
    'k-módů': (3, 5),  
    'CBRW': (1, 1),     
    'FPOF': (),     
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

    # plt.savefig(f'plots/{df_name}_silhouette_outliers_removed_diff_categorical.pdf', format='pdf')
    # plt.show()
    
# print(df_local.head())
# print(df_global.head())    
