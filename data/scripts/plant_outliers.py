import numpy as np
import pandas as pd
import random
import sys
sys.path.append('/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/data')

from scripts.get_data import get_data_numerical, get_data_categorical
from scripts.get_data_from_csv import get_data_from_csv, convert_iris_to_categorical
from scripts.data_transformations import split_df, concat_df

import warnings
warnings.filterwarnings('ignore')

IS_OUTLIER_COLUMN_NAME = 'IsOutlier'
SPECIES_COLUMN_NAME = 'Species'
FEATURE_NAMES = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
DIGITS_TO_ROUND = 4

def validate_percentage(percentage):
    if not 0 <= percentage <= 100:
        raise ValueError("Percentage must be between 0 and 100.")

def add_is_outlier_column(df):
    if IS_OUTLIER_COLUMN_NAME not in df.columns:
        df[IS_OUTLIER_COLUMN_NAME] = False

def get_columns_for_outliers(df, species_column, specified_columns=None):
    if specified_columns is None:
        return [col for col in df.select_dtypes(include=[np.number]).columns if col != species_column]
    else:
        return specified_columns

def get_number_of_outliers(df, percentage):
    return int(np.floor((percentage / 100) * len(df)))

def calculate_species_means(df, species_column):
    species_means = df.groupby(species_column).mean()
    return species_means.drop('IsOutlier', axis=1, errors='ignore')

def calculate_overall_means(df, columns):
    return df[columns].mean()

def calculate_outlier_value(df, column, rate):
    mean = df[column].mean()
    std_dev = df[column].std()
    direction = np.random.choice([-1, 1])
    outlier_value = max(0.01, mean + direction * rate * std_dev)
    return outlier_value

def create_local_outlier(species_means, chosen_species, columns, df, rate):
    species_mean_values = species_means.loc[chosen_species]
    new_row = species_mean_values.to_dict()

    outlier_column = np.random.choice(columns)
    outlier_value = calculate_outlier_value(df, outlier_column, rate)
    new_row[outlier_column] = round(outlier_value, DIGITS_TO_ROUND)
    new_row['IsOutlier'] = True

    return new_row

# def create_local_outlier(overall_means, columns, df, rate):
#     new_row = overall_means.to_dict()

#     outlier_column = np.random.choice(columns)
#     outlier_value = calculate_outlier_value(df, outlier_column, rate)
#     new_row[outlier_column] = round(outlier_value, DIGITS_TO_ROUND)
#     new_row[IS_OUTLIER_COLUMN_NAME] = True

#     return new_row

def create_global_outlier(species_means, chosen_species, columns, df, rate):
    species_mean_values = species_means.loc[chosen_species]
    new_row = species_mean_values.to_dict()

    for outlier_column in columns:
        outlier_value = calculate_outlier_value(df, outlier_column, rate)
        new_row[outlier_column] = round(outlier_value, DIGITS_TO_ROUND)

    new_row['IsOutlier'] = True

    return new_row

# def create_global_outlier(overall_means, columns, df, rate):
#     new_row = overall_means.to_dict()

#     for outlier_column in columns:
#         outlier_value = calculate_outlier_value(df, outlier_column, rate)
#         new_row[outlier_column] = round(outlier_value, DIGITS_TO_ROUND)

#     new_row[IS_OUTLIER_COLUMN_NAME] = True

#     return new_row

def add_local_outliers(df, outlier_percentage, rate, species_column=SPECIES_COLUMN_NAME, columns=None, add_species_outlier_info=True):
    validate_percentage(outlier_percentage)
    add_is_outlier_column(df)
    columns = get_columns_for_outliers(df, species_column, columns)
    species_means = calculate_species_means(df, species_column)

    num_outliers = get_number_of_outliers(df, outlier_percentage)

    for _ in range(num_outliers):
        chosen_species = np.random.choice(df[species_column].unique())
        new_row = create_local_outlier(species_means, chosen_species, columns, df, rate)
        new_row[species_column] = chosen_species
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    if add_species_outlier_info == True:
        df.loc[df[IS_OUTLIER_COLUMN_NAME] == True, species_column] = df[species_column] + ' (Lokální Outlier)'
    else:
        df.loc[df[IS_OUTLIER_COLUMN_NAME] == True, species_column] = df[species_column]

    return df

# def add_local_outliers(df, outlier_percentage, rate, species_column='Species', columns=None, add_species_outlier_info=True):
#     validate_percentage(outlier_percentage)
#     add_is_outlier_column(df)
    
#     if columns is None:
#         columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
#     overall_means = calculate_overall_means(df, columns)
    
#     num_outliers = get_number_of_outliers(df, outlier_percentage)

#     for _ in range(num_outliers):
#         new_row = create_local_outlier(overall_means, columns, df, rate)
        
#         if add_species_outlier_info:
#             new_row[species_column] = 'Lokální Outlier'

#         df = pd.concat([df, pd.DataFrame([new_row], index=[0])], ignore_index=True)

#     return df

def add_global_outliers(df, outlier_percentage, rate, species_column=SPECIES_COLUMN_NAME, columns=None):
    validate_percentage(outlier_percentage)
    add_is_outlier_column(df)
    columns = get_columns_for_outliers(df, species_column, columns)
    species_means = calculate_species_means(df, species_column)

    num_outliers = get_number_of_outliers(df, outlier_percentage)

    for _ in range(num_outliers):
        chosen_species = np.random.choice(df[species_column].unique())
        new_row = create_global_outlier(species_means, chosen_species, columns, df, rate)
        new_row[species_column] = chosen_species

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.loc[df[IS_OUTLIER_COLUMN_NAME] == True, species_column] = df[species_column] + ' (Globální Outlier)'
    return df

# def add_global_outliers(df, outlier_percentage, rate, species_column=SPECIES_COLUMN_NAME, columns=None):
#     validate_percentage(outlier_percentage)
#     add_is_outlier_column(df)
    
#     if columns is None:
#         columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
#     overall_means = calculate_overall_means(df, columns)
#     num_outliers = get_number_of_outliers(df, outlier_percentage)

#     for _ in range(num_outliers):
#         new_row = create_global_outlier(overall_means, columns, df, rate)
#         new_row[species_column] = 'Globální Outlier'

#         df = pd.concat([df, pd.DataFrame([new_row], index=[0])], ignore_index=True)

#     return df

def add_contextual_outliers(df, outlier_percentage, num_columns=4, species_column=SPECIES_COLUMN_NAME):
    validate_percentage(outlier_percentage)

    species_list = df[species_column].unique()
    features = FEATURE_NAMES
    
    outliers = pd.DataFrame(columns=df.columns)

    overall_min = df[features].min()
    overall_max = df[features].max()

    std_dev = df[features].std()
    
    for species in species_list:
        species_df = df[df[species_column] == species]
        species_means = species_df[features].mean()
        
        num_outliers_total = int(len(df) * (outlier_percentage / 100.0))
        num_outliers_per_species = max(1, num_outliers_total // len(species_list))
        
        for _ in range(num_outliers_per_species):
            outlier = species_means.copy()
            features_to_change = random.sample(FEATURE_NAMES, num_columns)
            for feature in features_to_change:
                deviation = np.random.uniform(0, 1) * std_dev[feature]
                if overall_max[feature] - species_means[feature] > species_means[feature] - overall_min[feature]:
                    outlier[feature] = round(overall_max[feature] - deviation, DIGITS_TO_ROUND)
                else:
                    outlier[feature] = round(overall_min[feature] + deviation, DIGITS_TO_ROUND)
            
            outlier[species_column] = species + ' (Kontextuální Outlier)'
            outlier_row = pd.DataFrame([outlier], columns=df.columns)
            outliers = pd.concat([outliers, outlier_row], ignore_index=True)
    
    df[IS_OUTLIER_COLUMN_NAME] = False
    outliers[IS_OUTLIER_COLUMN_NAME] = True
    
    df_with_outliers = pd.concat([df, outliers], ignore_index=True)
    
    return df_with_outliers

def add_collective_outliers(df, outlier_percentage, species_column=SPECIES_COLUMN_NAME):
    validate_percentage(outlier_percentage)
    features = FEATURE_NAMES
    overall_min = df[features].min()
    overall_max = df[features].max()
    std_dev = df[features].std()

    features_with_low_values = ['PetalLengthCm', 'PetalWidthCm']
    features_with_high_values = ['SepalLengthCm', 'SepalWidthCm']

    num_outliers = get_number_of_outliers(df, outlier_percentage)
    outlier_rows = []
    
    for _ in range(num_outliers):
        outlier = {}
        for feature in features:
            if feature in features_with_low_values:
                deviation = np.random.uniform(0, 1) * std_dev[feature]
                outlier[feature] = round(overall_min[feature] + deviation, DIGITS_TO_ROUND)
            elif feature in features_with_high_values:
                deviation = np.random.uniform(0, 1) * std_dev[feature]
                outlier[feature] = round(overall_max[feature] - deviation, DIGITS_TO_ROUND)
        
        outlier[species_column] = 'Kolektivní Outlier'
        outlier_rows.append(outlier)
    
    df['IsOutlier'] = False
    outliers = pd.DataFrame(outlier_rows, columns=df.columns)
    outliers['IsOutlier'] = True

    df_with_outliers = pd.concat([df, outliers], ignore_index=True)
    
    return df_with_outliers


# file_path_raw_data = '/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/data/numerical/Iris.csv'

# df = get_data_from_csv(file_path_raw_data)
# df_outliers = get_data_from_csv(file_path_outliers)

# df = concat_df(df_raw, df_outliers)
# X, y = split_df(df)
# df = df.drop('Id', axis=1)
# print(df.describe())
# print(X.head())
# print(X.shape)
# print(X['SepalLengthCm'].describe())
# print(max_sepal_length)
# X_withOutliers = plant_local_outliers(df, 10, 3, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
# print(X_withOutliers.head(30))

# if 'IsOutlier' not in df.columns:
#     df['IsOutlier'] = False


# Add outlier observations
# df_with_local_outliers = add_local_outliers(df, outlier_percentage=5, rate=3, species_column='Species')
# print(df_with_local_outliers.tail(10)) 
# df_with_global_outliers = add_global_outliers(df, outlier_percentage=5, rate=3, species_column='Species')
# print(df_with_global_outliers.tail(10))


# print(f'Means per species:\n{df.groupby("Species").mean()}')
# df_with_contextual_outliers = add_contextual_outliers(df, outlier_percentage=5, num_columns=2)
# print(df_with_contextual_outliers.describe())
# print(df_with_contextual_outliers.head(5))
# print(df_with_contextual_outliers.tail(10))
# print('Iris-setosa:\n' ,df.loc[df['Species'] == 'Iris-setosa'].describe())
# print('Iris-setosa:\n' ,df_with_contextual_outliers.loc[df_with_contextual_outliers['Species'] == 'Iris-setosa'].describe())

# print('Iris-versicolor:\n' ,df.loc[df['Species'] == 'Iris-versicolor'].describe())
# print('Iris-versicolor:\n' ,df_with_contextual_outliers.loc[df_with_contextual_outliers['Species'] == 'Iris-versicolor'].describe())

# print('Iris-virginica:\n' ,df.loc[df['Species'] == 'Iris-virginica'].describe())
# print('Iris-virginica:\n' ,df_with_contextual_outliers.loc[df_with_contextual_outliers['Species'] == 'Iris-virginica'].describe())

# print(df_with_outliers.head(30))
# print(df_with_local_outliers.tail(17))
# print(df_with_global_outliers.tail(17))



# df_with_collective_outliers = add_collective_outliers(df, 5)
# print(df_with_collective_outliers.tail(10))