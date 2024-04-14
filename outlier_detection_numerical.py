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


def detect_outliers_kmeans(X, percentage, k=3, random_state=1117):
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(X)
    clusters = kmeans.predict(X)

    X_dist = kmeans.transform(X)
    X_dist = np.min(X_dist, axis=1)

    threshold = np.percentile(X_dist, 100 - percentage*100)
    outliers = X_dist > threshold
    
    return outliers #, threshold

def detect_outliers_iforest(X, n_estimators=200, contamination=0.02, random_state=17):
    iForest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    iForest.fit(X)
    return [True if outlier == -1 else False for outlier in iForest.predict(X)]

def detect_outliers_lof(X, n_neighbors=7, contamination=0.01):
    LOF = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    return [True if outlier == -1 else False for outlier in LOF.fit_predict(X)]

def purity_score(y_true, y_pred):
    contingency_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

####################
# Data preparation #
####################
# file_path_raw_data = '/Users/ngocphuong.vu/skola/diplomka/code/Influence of Outliers on Clustering/data/Iris.csv'
# file_path_outliers = '/Users/ngocphuong.vu/skola/diplomka/code/Influence of Outliers on Clustering/data/Iris-artificial-outliers.csv'

# df_raw = get_data_from_csv(file_path_raw_data)
# df_outliers = get_data_from_csv(file_path_outliers)

# df = concat_df(df_raw, df_outliers)


file_path_raw_data = '/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/data/numerical/Iris-czech.csv'
SPECIES_COLUMN_NAME = 'Druh'
IS_OUTLIER_COLUMN_NAME = 'IsOutlier'
df = get_data_from_csv(file_path_raw_data)
df = df.drop('Id', axis=1)
X, y = split_df(df)

#########################
### Generate Outliers ###
#########################
# np.random.seed(19)
# ### Local outliers
# df_with_local_outliers_1percent = add_local_outliers(df, outlier_percentage=1, rate=3.5, species_column=SPECIES_COLUMN_NAME)
# df_with_local_outliers_5percent = add_local_outliers(df, outlier_percentage=5, rate=3.5, species_column=SPECIES_COLUMN_NAME)
# df_with_local_outliers_10percent = add_local_outliers(df, outlier_percentage=10, rate=3.5, species_column=SPECIES_COLUMN_NAME)

# X_local_outliers_1percent, y_local_outliers_1percent = split_df(df_with_local_outliers_1percent, number_of_columns=2)
# X_local_outliers_5percent, y_local_outliers_5percent = split_df(df_with_local_outliers_5percent, number_of_columns=2)
# X_local_outliers_10percent, y_local_outliers_10percent = split_df(df_with_local_outliers_10percent, number_of_columns=2)

# print(y_local_outliers_1percent)

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

# # print(df_with_local_outliers.head(5))
# print(df_with_local_outliers_10percent.tail(15))
# print(df_with_global_outliers_5percent.tail(15))
# print(df_with_contextual_outliers_1percent.tail(15))
# print(df_with_collective_outliers_10percent.tail(15))

###################################
# Cluster based Outlier Detection #
###################################
###
### K-means
###
# local_outliers_kMeans_1percent = detect_outliers_kmeans(X_local_outliers_1percent, percentage=0.01)
# local_outliers_kMeans_5percent = detect_outliers_kmeans(X_local_outliers_5percent, percentage=0.05)
# local_outliers_kMeans_10percent = detect_outliers_kmeans(X_local_outliers_10percent, percentage=0.1)

# global_outliers_kMeans_1percent = detect_outliers_kmeans(X_global_outliers_1percent, percentage=0.01)
# global_outliers_kMeans_5percent = detect_outliers_kmeans(X_global_outliers_5percent, percentage=0.05)
# global_outliers_kMeans_10percent = detect_outliers_kmeans(X_global_outliers_10percent, percentage=0.1)

# contextual_outliers_kMeans_1percent = detect_outliers_kmeans(X_contextual_outliers_1percent, percentage=0.01)
# contextual_outliers_kMeans_5percent = detect_outliers_kmeans(X_contextual_outliers_5percent, percentage=0.05)
# contextual_outliers_kMeans_10percent = detect_outliers_kmeans(X_contextual_outliers_10percent, percentage=0.1)

# collective_outliers_kMeans_1percent = detect_outliers_kmeans(X_collective_outliers_1percent, percentage=0.01)
# collective_outliers_kMeans_5percent = detect_outliers_kmeans(X_collective_outliers_5percent, percentage=0.05)
# collective_outliers_kMeans_10percent = detect_outliers_kmeans(X_collective_outliers_10percent, percentage=0.1)


###
### DBScan
###
# dbscan = DBSCAN(eps=2.8, min_samples=4) # Trial and error

# # Fit the model
# dbscan.fit(X_local_outliers)

# # Outliers labelled with -1, different clusters get non-negative integers
# outliers_DBScan = dbscan.labels_ == -1

####################
# Isolation Forest #
####################
local_outliers_iForest_1percent = detect_outliers_iforest(X_local_outliers_1percent, contamination=0.01)
local_outliers_iForest_5percent = detect_outliers_iforest(X_local_outliers_5percent, contamination=0.05)
local_outliers_iForest_10percent = detect_outliers_iforest(X_local_outliers_10percent, contamination=0.1)

global_outliers_iForest_1percent = detect_outliers_iforest(X_global_outliers_1percent, contamination=0.01)
global_outliers_iForest_5percent = detect_outliers_iforest(X_global_outliers_5percent, contamination=0.05)
global_outliers_iForest_10percent = detect_outliers_iforest(X_global_outliers_10percent, contamination=0.1)

contextual_outliers_iForest_1percent = detect_outliers_iforest(X_contextual_outliers_1percent, contamination=0.01, n_estimators=100)
contextual_outliers_iForest_5percent = detect_outliers_iforest(X_contextual_outliers_5percent, contamination=0.05, n_estimators=100)
contextual_outliers_iForest_10percent = detect_outliers_iforest(X_contextual_outliers_10percent, contamination=0.1, n_estimators=100)

collective_outliers_iForest_1percent = detect_outliers_iforest(X_collective_outliers_1percent, contamination=0.01, n_estimators=100)
collective_outliers_iForest_5percent = detect_outliers_iforest(X_collective_outliers_5percent, contamination=0.05, n_estimators=100)
collective_outliers_iForest_10percent = detect_outliers_iforest(X_collective_outliers_10percent, contamination=0.1, n_estimators=100)

########################
# Local Outlier Factor #
########################
local_outliers_LOF_1percent = detect_outliers_lof(X_local_outliers_1percent, contamination=0.01, n_neighbors=7)
local_outliers_LOF_5percent = detect_outliers_lof(X_local_outliers_5percent, contamination=0.05, n_neighbors=7)
local_outliers_LOF_10percent = detect_outliers_lof(X_local_outliers_10percent, contamination=0.1, n_neighbors=7)

global_outliers_LOF_1percent = detect_outliers_lof(X_global_outliers_1percent, contamination=0.01, n_neighbors=17)
global_outliers_LOF_5percent = detect_outliers_lof(X_global_outliers_5percent, contamination=0.05, n_neighbors=17)
global_outliers_LOF_10percent = detect_outliers_lof(X_global_outliers_10percent, contamination=0.1, n_neighbors=11)

contextual_outliers_LOF_1percent = detect_outliers_lof(X_contextual_outliers_1percent, contamination=0.01, n_neighbors=7)
contextual_outliers_LOF_5percent = detect_outliers_lof(X_contextual_outliers_5percent, contamination=0.05, n_neighbors=7)
contextual_outliers_LOF_10percent = detect_outliers_lof(X_contextual_outliers_10percent, contamination=0.1, n_neighbors=7)

collective_outliers_LOF_1percent = detect_outliers_lof(X_collective_outliers_1percent, contamination=0.01, n_neighbors=17)
collective_outliers_LOF_5percent = detect_outliers_lof(X_collective_outliers_5percent, contamination=0.05, n_neighbors=27)
collective_outliers_LOF_10percent = detect_outliers_lof(X_collective_outliers_10percent, contamination=0.1, n_neighbors=37)


def print_outlier_results():
    methods = ['kMeans', 'iForest', 'LOF']
    types = ['local', 'global', 'contextual', 'collective']
    percentages = [0.01, 0.05, 0.1]

    for method in methods:
        print(f"\nMethod: {method}")
        for type in types:
            print(f"\nType: {type}")
            for percentage in percentages:
                variable_name = f"{type}_outliers_{method}_"+str(int(percentage*100))+"percent"
                conf_matrix = confusion_matrix(globals()[variable_name], globals()[f"y_{type}_outliers_"+str(int(percentage*100))+"percent"][IS_OUTLIER_COLUMN_NAME])
                print(f"Percentage: {percentage*100}%")
                print(conf_matrix)
                print(f'True Positive: {conf_matrix[1][1]}, True Negative: {conf_matrix[0][0]}, False Positive: {conf_matrix[0][1]}, False Negative: {conf_matrix[1][0]}')
                print(f'True Positive Rate: {conf_matrix[1][1]/(conf_matrix[1][1]+conf_matrix[1][0])}')
                # print(f"Labels: {globals()[variable_name]}")

print_outlier_results()


import matplotlib.lines as mlines

# def plot_tpr_vs_percentage_for_types():
#     methods_labels = ['shlukování metodou k-průměrů', 'iForest', 'LOF']
#     methods = ['kMeans', 'iForest', 'LOF']
#     types = ['local', 'global', 'contextual', 'collective']
#     types_labels = ['Lokální', 'Globální', 'Kontextuální', 'Kolektivní']
#     percentages = [5, 10, 15]

#     # Plot style setup
#     markers = ['o', 's', '^', 'D']
#     colors = ['b', 'g', 'r', 'c']
    
#     fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
#     axs = axs.ravel()

#     for type_idx, type in enumerate(types):
#         for method_idx, method in enumerate(methods):
#             tprs = np.random.rand(3)  # Example TPR data
#             axs[type_idx].plot(percentages, tprs, marker=markers[method_idx], color=colors[method_idx], label=methods_labels[method_idx])
#             axs[type_idx].set_title(f'{types_labels[type_idx]} odlehlé hodnoty')
#             axs[type_idx].set_xlabel('Procento odlehlých hodnot')
#             axs[type_idx].set_ylabel('Sensitivita (TPR)')
#             axs[type_idx].set_xticks(percentages)
#             axs[type_idx].grid(True)

#     legend_handles = [mlines.Line2D([], [], color=colors[i], marker=markers[i], linestyle='-', label=label) for i, label in enumerate(methods_labels)]
#     plt.legend(handles=legend_handles, loc='upper center', ncol=3, fancybox=True, shadow=True)

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     # plt.suptitle('Sensitivita metody pro různé typy odlehlých hodnot', fontsize=16)
#     plt.show()



def plot_tpr_vs_percentage_for_types(save=False, file_name=None):
    methods_labels = ['shlukování metodou k-průměrů', 'iForest', 'LOF']
    methods = ['kMeans', 'iForest', 'LOF']
    types = ['local', 'global', 'contextual', 'collective']
    types_labels = ['lokální', 'globální', 'kontextuální', 'kolektivní']
    percentages = [1, 5, 10]
    markers = ['o', 's', '^', 'D']
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    for type_idx, type in enumerate(types):
        for method_idx, method in enumerate(methods):
            tprs = []
            for percentage in percentages:
                variable_name = f"{type}_outliers_{method}_"+str(int(percentage))+"percent"
                conf_matrix = confusion_matrix(globals()[variable_name], globals()[f"y_{type}_outliers_"+str(int(percentage))+"percent"][IS_OUTLIER_COLUMN_NAME])
                TPR = conf_matrix[1][1]/(conf_matrix[1][1]+conf_matrix[1][0])
                tprs.append(TPR)
                
            axs[type_idx].plot(percentages, tprs, marker=markers[method_idx % len(markers)], label=methods_labels[method_idx])
            axs[type_idx].set_title(f'{types_labels[type_idx].capitalize()} odlehlé hodnoty')
            axs[type_idx].set_xlabel('Procento odlehlých hodnot')
            axs[type_idx].set_ylabel('Sensitivita (TPR)')
            axs[type_idx].set_ylim([0, 1])
            axs[type_idx].set_xticks(percentages)
            axs[type_idx].grid(True)
            axs[type_idx].legend(loc='lower right')

    plt.tight_layout()
    plt.suptitle('Sensitivita pro různé metody a typy odlehlých hodnot', y=1.02)
    if save:
        plt.savefig(f'{file_name}.pdf', format='pdf')
    else:
        plt.show()

def plot_tnr_vs_percentage_for_types(save=False, file_name=None):
    methods_labels = ['shlukování metodou k-průměrů', 'iForest', 'LOF']
    methods = ['kMeans', 'iForest', 'LOF']
    types = ['local', 'global', 'contextual', 'collective']
    types_labels = ['lokální', 'globální', 'kontextuální', 'kolektivní']
    percentages = [1, 5, 10]

    markers = ['o', 's', '^', 'D']
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    for type_idx, type in enumerate(types):
        for method_idx, method in enumerate(methods):
            tnrs = []
            for percentage in percentages:
                variable_name = f"{type}_outliers_{method}_"+str(int(percentage))+"percent"
                conf_matrix = confusion_matrix(globals()[variable_name], globals()[f"y_{type}_outliers_"+str(int(percentage))+"percent"][IS_OUTLIER_COLUMN_NAME])
                TNR = conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[0][1])
                tnrs.append(TNR)
                
            axs[type_idx].plot(percentages, tnrs, marker=markers[method_idx % len(markers)], label=methods_labels[method_idx])
            axs[type_idx].set_title(f'{types_labels[type_idx].capitalize()} odlehlé hodnoty')
            axs[type_idx].set_xlabel('Procento odlehlých hodnot')
            axs[type_idx].set_ylabel('Specificita (TNR)')
            axs[type_idx].set_ylim([0, 1])
            axs[type_idx].set_xticks(percentages)
            axs[type_idx].grid(True)
            axs[type_idx].legend(loc='lower right')

    plt.tight_layout()
    plt.suptitle('Specificita pro různé metody a typy odlehlých hodnot', y=1.02)
    if save:
        plt.savefig(f'{file_name}.pdf', format='pdf')
    else:
        plt.show()

# plot_tpr_vs_percentage_for_types()
# plot_tnr_vs_percentage_for_types()
        
# plot_tpr_vs_percentage_for_types(save=True, file_name='tpr_vs_percentage_for_types')
# plot_tnr_vs_percentage_for_types(save=True, file_name='tnr_vs_percentage_for_types')


##################
### Clustering ###
##################
n_cluster = 3
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f'y: {y}')
print(f'y_encoded: {y_encoded}')

# Initialize models
linkages = ['ward', 'complete', 'average', 'single']
metrics = ['euclidean', 'manhattan']

kmeans_model = KMeans(n_clusters=n_cluster, random_state=42)
kmeans_labels = kmeans_model.fit_predict(X)
kmeans_ari_score = adjusted_rand_score(y_encoded, kmeans_labels)
kmeans_purity = purity_score(y_encoded, kmeans_labels)
# print(f'K-means ARI Score: {kmeans_ari_score}, Purity: {kmeans_purity}')

# for linkage_type in linkages:
#     for metric_type in metrics:
#         if linkage_type == 'ward' and metric_type != 'euclidean':
#             continue
        
#         model = AgglomerativeClustering(n_clusters=n_cluster, metric=metric_type, linkage=linkage_type)
#         labels = model.fit_predict(X)
#         ari_score = adjusted_rand_score(y_encoded, labels)
#         purity = purity_score(y_encoded, labels)
#         # print(f'Labels for {metric_type.title()} {linkage_type.title()}: {labels}')
#         print(f'Linkage: {linkage_type.title()}, Metric: {metric_type.title()}, ARI Score: {ari_score}, Purity: {purity}')


# dataset_with_outliers = [
#     X_local_outliers_1percent, X_local_outliers_5percent, X_local_outliers_10percent,
#     X_global_outliers_1percent, X_global_outliers_5percent, X_global_outliers_10percent,
#     X_contextual_outliers_1percent, X_contextual_outliers_5percent, X_contextual_outliers_10percent,
#     X_collective_outliers_1percent, X_collective_outliers_5percent, X_collective_outliers_10percent]
# for data_with_outliers in dataset_with_outliers:
#     for linkage_type in linkages:
#         for metric_type in metrics:
#             if linkage_type == 'ward' and metric_type != 'euclidean':
#                 continue
            
#             model = AgglomerativeClustering(n_clusters=n_cluster, metric=metric_type, linkage=linkage_type)
#             labels = model.fit_predict(data_with_outliers)
#             ari_score = adjusted_rand_score(y_encoded, labels)
#             purity = purity_score(y_encoded, labels)
#             # print(f'Labels for {metric_type.title()} {linkage_type.title()}: {labels}')
#             print(f'\
#                   Dataset: {data_with_outliers},\
#                   Linkage: {linkage_type.title()},\
#                   Metric: {metric_type.title()},\
#                   ARI Score: {ari_score},\
#                   Purity: {purity}')

model_euclidean_ward = AgglomerativeClustering(n_clusters=n_cluster, metric='euclidean', linkage='ward')
model_euclidean_ward = model_euclidean_ward.fit(X)
print(f'Labels Euclidean Ward: {model_euclidean_ward.labels_}')

# model_euclidean_average = AgglomerativeClustering(n_clusters=n_cluster, metric='euclidean', linkage='average')
# model_euclidean_average = model_euclidean_average.fit(X)

# model_manhattan_complete = AgglomerativeClustering(n_clusters=n_cluster, metric='manhattan', linkage='complete')
# model_manhattan_complete = model_manhattan_complete.fit(X)

# Assign labels to each point
# labels_euclidean_ward = model_euclidean_ward.labels_
# labels_euclidean_average = model_euclidean_average.labels_
# labels_manhattan_complete = model_manhattan_complete.labels_

# label_encoder = LabelEncoder()
# true_labels_encoded = label_encoder.fit_transform(y)
# ari_score = adjusted_rand_score(true_labels_encoded, labels_euclidean_ward)

# print(f'Labels Euclidean Ward: {labels_euclidean_ward}')
# print(f'Labels Euclidean Average: {labels_euclidean_average}')
# print(f'Labels Manhattan Complete: {labels_manhattan_complete}')

# print(f'Adjusted Rand Index: {ari_score}')
        

# # Scipy
# # Perform hierarchical/agglomerative clustering
# Z = linkage(X, 'ward')  # 'ward' is one method of hierarchical clustering

# # Plot the dendrogram
# plt.figure(figsize=(12, 5))
# dendrogram(Z)
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Sample index')
# plt.ylabel('Distance')
# plt.show()
