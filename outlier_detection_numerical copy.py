import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, ConfusionMatrixDisplay

from data.scripts.get_data import get_data_numerical, get_data_categorical
from data.scripts.get_data_from_csv import get_data_from_csv, convert_iris_to_categorical
from data.scripts.data_transformations import split_df, concat_df
from data.scripts.plant_outliers import add_local_outliers, add_global_outliers, add_contextual_outliers, add_collective_outliers


def detect_outliers_kmeans(X, contamination, k=3, random_state=1117):
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(X)
    clusters = kmeans.predict(X)

    X_dist = kmeans.transform(X)
    X_dist = np.min(X_dist, axis=1)

    threshold = np.percentile(X_dist, 100 - contamination*100)
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

outlier_detection_methods = {
    'KMeans': detect_outliers_kmeans,
    'IsolationForest': detect_outliers_iforest,
    'LocalOutlierFactor': detect_outliers_lof
}

for name, df in data_wOutliers.items():
    # if name not in ['df_local_outliers_10percent']:
    #     continue
    for method_name, method in outlier_detection_methods.items():
        X, y = split_df(df, number_of_columns=2)
        percentage = int(name.split('_')[-1].replace('percent', '')) / 100 - 0.009
        # print(percentage)
        outliers = method(X, contamination=percentage)
        df_no_outliers = df[~np.array(outliers)]
        # print(df_no_outliers)
        df_no_outliers.to_csv(f'data/numerical/wOutliers/run1/removed/{method_name}/{name}_removed_{method_name}.csv', index=False)

        # Compare outliers with y[0] and compute metrics
        true_outliers = y['IsOutlier']
        cm = confusion_matrix(true_outliers, outliers)
        accuracy = accuracy_score(true_outliers, outliers)
        precision = precision_score(true_outliers, outliers)
        recall = recall_score(true_outliers, outliers)
        f1 = f1_score(true_outliers, outliers)

        
        # ConfusionMatrixDisplay(cm).plot()

        print(f'Outliers for {name} using {method_name}:') #{outliers}')
        print(f'Outliers for {name} using {method_name}: {outliers}')
        print(f'Confusion matrix:\n {cm}')
        print(f'{percentage}\\% & \\{method_name} & {accuracy * 100:.2f} & {precision * 100:.2f} & {recall * 100:.2f} & {f1 * 100:.2f} \\\\')
        
        cm_df = pd.DataFrame(cm, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive'])
        print(tabulate(cm_df, headers='keys', tablefmt='psql'))
        # print(f'Accuracy: {accuracy:.4f}')
        # print(f'Precision: {precision:.4f}')
        # print(f'Recall: {recall:.4f}')
        # print(f'F1 score: {f1:.4f}\n')
