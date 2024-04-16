import numpy as np
import pandas as pd
import gower
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

def hamming_distance(u, v):
    return np.sum(u != v) / len(u)

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

# Create a copy of the dataframe for the qcut method
df_qcut = df.copy()

df_cut = df.copy()

df_columns = df.columns
for column in df_columns[:-1]:
    df_cut[column] = pd.cut(df_cut[column], bins=NB_BINS)
    print(df_cut[column].value_counts())

# for column in df_columns[:-1]:
#     df_qcut[column] = pd.qcut(df_qcut[column], q=NB_BINS)
#     print(df_qcut[column].value_counts())

print(df_cut.dtypes)
print(df_cut.head())
# print(df_qcut.head())

X_cut, y_cut = split_df(df_cut)
# X_qcut, y_qcut = split_df(df_qcut)

n_clusters = 3

kmodes_cut = KModes(n_clusters=n_clusters, random_state=42)
clusters_cut = kmodes_cut.fit_predict(X_cut)

# kmodes_qcut = KModes(n_clusters=n_clusters, random_state=42)
# clusters_qcut = kmodes_qcut.fit_predict(X_qcut)

print(f'Clusters for cut data: {clusters_cut}')
# print(f'Clusters for qcut data: {clusters_qcut}')

# Convert all columns to string
X_cut = X_cut.astype(str)
# X_qcut = X_qcut.astype(str)

print(X_cut.dtypes)
# print(X_qcut.dtypes)

def jaccard_similarity(x, y):
    set_x = set(x)
    set_y = set(y)
    intersection = len(set_x & set_y)
    union = len(set_x | set_y)
    return intersection / union

dist_matrix_jaccard_cut = squareform(pdist(X_cut, lambda u, v: 1 - jaccard_similarity(u, v)))
# dist_matrix_jaccard_qcut = squareform(pdist(X_qcut, lambda u, v: 1 - jaccard_similarity(u, v)))
# print(dist_matrix_jaccard_cut)
dist_matrix_hamming_cut = squareform(pdist(X_cut, metric=hamming_distance))
dist_matrix_smc_cut = squareform(pdist(X_cut, metric=smc_distance))

distance_matrix_gower_cut = gower.gower_matrix(X_cut)
# distance_matrix_gower_qcut = gower.gower_matrix(X_qcut)

print(f'Gower distance matrix cut: {distance_matrix_gower_cut}, \n \
      Hamming distance matrix cut: {dist_matrix_hamming_cut}, \n \
        SMC distance matrix cut: {dist_matrix_smc_cut}')
linkages = ['single', 'average', 'complete']
metrics = ['gower', 'hamming', 'smc', 'jaccard']
dist_matrices = {
    'gower': distance_matrix_gower_cut,
    'hamming': dist_matrix_hamming_cut,
    'smc': dist_matrix_smc_cut,
    'jaccard': dist_matrix_jaccard_cut
}
data = {}

for metric in metrics:
    for linkage in linkages:
        clusters = AgglomerativeClustering(n_clusters=NB_CLUSTERS, linkage=linkage, metric='precomputed').fit_predict(dist_matrices[metric])
        data[f'Hierarchical {linkage.capitalize()} {metric.capitalize()}'] = (dist_matrices[metric], clusters)
        # print(f'Clusters for hierarchical {linkage} cut {metric}: {clusters}')

    data[f'K-modes {metric.capitalize()}'] = (dist_matrices[metric], clusters_cut)

for name, (dist_matrix, clusters) in data.items():
    silhouette_index = silhouette_score(dist_matrix, clusters, metric='precomputed')
    dbi_index = calculate_dbi(dist_matrix, clusters)
    print(f'{name} & {silhouette_index:.4f} & {dbi_index:.4f} \\\\')

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
    return clusters, silhouette_index, dbi_index

linkage_names = {'single': '\\nearestNeighbourMethod', 'average': '\\averageLinkageMethod', 'complete': '\\farthestNeighbourMethod'}
kmodes_name = '\\kModesMethod'

linkages = ['single', 'average', 'complete']

for name, df in data_wOutliers.items():
    print(f'Processing {name}')
    percentage = name.split('_')[-1].replace('percent', '')
    X, y = split_df(df, number_of_columns=2)
    X = X.astype(str)
    for column in X.columns:
        print(X[column].value_counts())

    distance_matrix_smc = squareform(pdist(X, metric=smc_distance))

    for linkage in linkages:
        clusters, silhouette_index, dbi_index = perform_clustering_and_scoring(X, linkage, distance_matrix_smc)
        print(f'{percentage}\\% & {linkage_names[linkage]} & {silhouette_index:.4f} & {dbi_index:.4f} \\\\')


    kmodes = KModes(n_clusters=NB_CLUSTERS)
    clusters_kmodes = kmodes.fit_predict(X)
    silhouette_index_kmodes = silhouette_score(distance_matrix_smc, clusters_kmodes, metric='precomputed')
    dbi_index_kmodes = calculate_dbi(distance_matrix_smc, clusters_kmodes)
    print(f' & {kmodes_name} & {silhouette_index_kmodes:.4f} & {dbi_index_kmodes:.4f} \\\\')

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
        clusters, silhouette_index, dbi_index = perform_clustering_and_scoring(X, linkage, distance_matrix_smc)
        print(f'{percentage}\\% & {linkage_names[linkage]} & {silhouette_index:.4f} & {dbi_index:.4f} \\\\')

    kmodes = KModes(n_clusters=NB_CLUSTERS)
    clusters_kmodes = kmodes.fit_predict(X)
    silhouette_index_kmodes = silhouette_score(distance_matrix_smc, clusters_kmodes, metric='precomputed')
    dbi_index_kmodes = calculate_dbi(distance_matrix_smc, clusters_kmodes)
    print(f' & {kmodes_name} & {silhouette_index_kmodes:.4f} & {dbi_index_kmodes:.4f} \\\\')

for name, df in data_wOutliers_removed_CBRW.items():
    print(f'\n\nProcessing {name}')
    percentage = name.split('_')[-3].replace('percent', '')
    X, y = split_df(df, number_of_columns=2)
    X = X.astype(str)
    distance_matrix_smc = squareform(pdist(X, metric=smc_distance))

    for linkage in linkages:
        clusters, silhouette_index, dbi_index = perform_clustering_and_scoring(X, linkage, distance_matrix_smc)
        print(f'{percentage}\\% & {linkage_names[linkage]} & {silhouette_index:.4f} & {dbi_index:.4f} \\\\')

    kmodes = KModes(n_clusters=NB_CLUSTERS)
    clusters_kmodes = kmodes.fit_predict(X)
    silhouette_index_kmodes = silhouette_score(distance_matrix_smc, clusters_kmodes, metric='precomputed')
    dbi_index_kmodes = calculate_dbi(distance_matrix_smc, clusters_kmodes)
    print(f' & {kmodes_name} & {silhouette_index_kmodes:.4f} & {dbi_index_kmodes:.4f} \\\\')

for name, df in data_wOutliers_removed_FPOF.items():
    print(f'\n\nProcessing {name}')
    percentage = name.split('_')[-3].replace('percent', '')
    X, y = split_df(df, number_of_columns=2)
    X = X.astype(str)
    distance_matrix_smc = squareform(pdist(X, metric=smc_distance))
    # print(X.head())

    for linkage in linkages:
        clusters, silhouette_index, dbi_index = perform_clustering_and_scoring(X, linkage, distance_matrix_smc)
        print(f'{percentage}\\% & {linkage_names[linkage]} & {silhouette_index:.4f} & {dbi_index:.4f} \\\\')

    kmodes = KModes(n_clusters=NB_CLUSTERS)
    clusters_kmodes = kmodes.fit_predict(X)
    silhouette_index_kmodes = silhouette_score(distance_matrix_smc, clusters_kmodes, metric='precomputed')
    dbi_index_kmodes = calculate_dbi(distance_matrix_smc, clusters_kmodes)
    print(f' & {kmodes_name} & {silhouette_index_kmodes:.4f} & {dbi_index_kmodes:.4f} \\\\')