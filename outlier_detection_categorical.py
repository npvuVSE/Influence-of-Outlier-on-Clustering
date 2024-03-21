import numpy as np
import os
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN

from data.get_data_from_csv import get_data_from_csv
from data_transformations import split_df
from algorithms.quantitative.cbrw import CBRW
from algorithms.quantitative.fpof import FPOF


def round_dict_values(input_dict, digits=4):
    """ Helper function for printing dicts with float values """
    return {key: round(val, digits) for key, val in input_dict.items()}

DATA_PATH = './data/qualitative/cheat.csv'

########
# FPOF #
########
df = get_data_from_csv(DATA_PATH)
X, y = split_df(df)

fpof_values, top_n_transactions, top_k_contradict_patterns = FPOF(X, min_support=0.3, top_n=4, top_k=3)
print('\nFPOF Values:')
for i, score in enumerate(fpof_values):
    print(f'Observation ID {i+1}: {round(score, 4)}')

print('\nTop-n transactions:')
for i, transaction in enumerate(top_n_transactions):
    print(f'Top-{i+1} transaction: {transaction}')

print('\nTop-k contradict patterns:')
for key, patterns in top_k_contradict_patterns.items():
    print(f'Top-{key} contradict patterns: {patterns}')


########
# CBRW #
########
# file_dir = os.path.abspath(os.path.dirname(__file__))
# DATA_PATH = os.path.join(file_dir, 'data/qualitative', 'CBRW_paper_example.csv')
# EXCLUDE_COLS = ['Cheat?']
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
print(f'Detector fit with {len(observations)} observations:')
for i, obs in enumerate(observations):
    print(f'Observation ID {i+1}: {obs}')

print('\nFeature weights:')
print(round_dict_values(detector.feature_weights, 4))

print('\nScores:')
for i, score in enumerate(scores):
    print(f'Observation ID {i+1}: {round(score, 4)}')

print('\nValue scores per attribute:')
for i, value_score in enumerate(value_scores):
    print(f'Observation ID {i+1}: {round_dict_values(value_score, 4)}')
