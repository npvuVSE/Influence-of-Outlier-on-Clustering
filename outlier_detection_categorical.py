import numpy as np
import os
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.cluster import KMeans, DBSCAN
from scipy.io.arff import loadarff 

from data.get_data_from_csv import get_data_from_csv
from data_transformations import split_df
from algorithms.quantitative.cbrw import CBRW
from algorithms.quantitative.fpof import FPOF


################
# Data loading #
################
### Cheat dataset
def round_dict_values(input_dict, digits=4):
    """ Helper function for printing dicts with float values """
    return {key: round(val, digits) for key, val in input_dict.items()}

DATA_PATH_CHEAT = './data/qualitative/cheat.csv'

# df = get_data_from_csv(DATA_PATH_CHEAT)
# X, y = split_df(df)

### Breat cancer
# fetch dataset
breast_cancer = fetch_ucirepo(id=14)
  
# data (as pandas dataframes)
X = breast_cancer.data.features
y = breast_cancer.data.targets

# print(X['inv-nodes'].value_counts())
# print(y.value_counts())

# print(y[y == 'recurrence-events'].index)
  
mapping_deg_malig = {1:'one',2:'two',3:'three'}
mapping_inv_nodes = {'0-2':'0-2','5-Mar':'3-5','8-Jun':'6-8','11-Sep':'9-11','14-Dec':'12-14','15-17':'15-17','18-20':'18-20','21-23':'21-23','24-26':'24-26','27-29':'27-29','30-32':'30-32','33-35':'33-35','36-39':'36-39'}
X['deg-malig'] = X['deg-malig'].map(mapping_deg_malig)
X['inv-nodes'] = X['inv-nodes'].map(mapping_inv_nodes)
X = X.astype(str)


# Combine X and y into one DataFrame
df = pd.concat([X, y], axis=1)

# Define the condition for y
condition = (df['Class'] == 'recurrence-events')  # replace 1 with the condition you want

df = df[df['node-caps'] != 'nan']
# Make dataset suitable for outlier detection
drop_indices = df[condition].sample(frac=0.93).index
df = df.drop(drop_indices)
df = df.reset_index(drop=True)

# Split df back into X and y
X = df.drop('Class', axis=1)
y = df['Class']

print(f"True outliers\n: {df[df['Class'] == 'recurrence-events']}")
# print(X.tail(6))
# print(X['inv-nodes'].value_counts())
# print(X['irradiat'].value_counts())

# print(X.shape)
# print(y.shape)
# print(X.head())
# print(y.value_counts())

# print(y[y == 'recurrence-events'].index)
# ########
# # FPOF #
# ########
fpof_values, top_n_transactions, top_k_contradict_patterns = FPOF(X, min_support=0.35, top_n=20, top_k=3)
# print('\nFPOF Values:')
# for i, score in enumerate(fpof_values):
#     print(f'Observation ID {i+1}: {round(score, 4)}')

# print('\nTop-n transactions:')
# for i, transaction in enumerate(top_n_transactions):
#     print(f'Top-{i+1} transaction (ID:): {transaction}')

# print('\nTop-k contradict patterns:')
# for key, patterns in top_k_contradict_patterns.items():
#     print(f'Top-{key} contradict patterns: {patterns}')


# ########
# # CBRW #
# ########
detector = CBRW()

# load data and add to detector as observations
# observations = load_from_csv(DATA_PATH, exclude_cols=EXCLUDE_COLS)
observations = X.to_dict('records')

# add observations to detector and fit
detector.add_observations(observations)
detector.fit()

# compute scores
scores = detector.score(observations)
value_scores = detector.value_scores(observations)

# display results
# print(f'Detector fit with {len(observations)} observations:')
# for i, obs in enumerate(observations):
#     print(f'Observation ID {i+1}: {obs}')

print('\nFeature weights:')
print(round_dict_values(detector.feature_weights, 4))

# print('\nScores:')
# for i, score in enumerate(scores):
#     print(f'Observation ID {i+1}: {round(score, 4)}')

# Create a list of tuples (id, score)
id_score_list = [(i+1, round(score, 4)) for i, score in enumerate(scores)]

# Sort the list in descending order by score
id_score_list.sort(key=lambda x: x[1], reverse=True)

# Print the top n scores
n = 20  # replace with the desired number
print('\nTop', n, 'scores:')
for i in range(n):
    print(f'Observation ID {id_score_list[i][0]}: {id_score_list[i][1]}')

# print('\nValue scores per attribute:')
# for i, value_score in enumerate(value_scores):
#     print(f'Observation ID {i+1}: {round_dict_values(value_score, 4)}')
