import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from data.scripts.get_data import get_data_numerical, get_data_categorical
from data.scripts.get_data_from_csv import get_data_from_csv, convert_iris_to_categorical
from data.scripts.data_transformations import split_df, concat_df
from data.scripts.plant_outliers import add_local_outliers, add_global_outliers, add_contextual_outliers, add_collective_outliers
from data.scripts.data_visualization import plot_species_boxplots, plot_pairwise_distributions

file_path_raw_data = '/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/data/numerical/Iris-czech.csv'
SPECIES_COLUMN_NAME = 'Druh'
IS_OUTLIER_COLUMN_NAME = 'IsOutlier'
df = get_data_from_csv(file_path_raw_data)
df = df.drop('Id', axis=1)
# print(df.head())

np.random.seed(117)
df_with_local_outliers = add_local_outliers(df, outlier_percentage=5, rate=3, species_column=SPECIES_COLUMN_NAME, add_species_outlier_info=True)
df_with_global_outliers = add_global_outliers(df, outlier_percentage=5, rate=3, species_column=SPECIES_COLUMN_NAME)
df_with_contextual_outliers = add_contextual_outliers(df, outlier_percentage=5, num_columns=3, species_column=SPECIES_COLUMN_NAME)
df_with_collective_outliers = add_collective_outliers(df, 5, species_column=SPECIES_COLUMN_NAME)

# print(df_with_local_outliers.head(5))
print(df_with_local_outliers.tail(15))
print(df_with_global_outliers.tail(15))
print(df_with_contextual_outliers.tail(15))
print(df_with_collective_outliers.tail(15))

hue_order_contextual = ['Iris-setosa', 'Iris-setosa (Kontextuální Outlier)', 'Iris-versicolor', 'Iris-versicolor (Kontextuální Outlier)', 'Iris-virginica', 'Iris-virginica (Kontextuální Outlier)']
hue_order_local = ['Iris-setosa', 'Iris-setosa (Lokální Outlier)', 'Iris-versicolor', 'Iris-versicolor (Lokální Outlier)', 'Iris-virginica', 'Iris-virginica (Lokální Outlier)']
hue_order_global = ['Iris-setosa', 'Iris-setosa (Globální Outlier)', 'Iris-versicolor', 'Iris-versicolor (Globální Outlier)', 'Iris-virginica', 'Iris-virginica (Globální Outlier)']

# plot_species_boxplots(df)
plot_pairwise_distributions(
    df,
    species_column=SPECIES_COLUMN_NAME,
    save_as_pdf=True,
    file_name='iris_pairwise_distributions'
    )
plot_pairwise_distributions(
    df_with_local_outliers,
    species_column=SPECIES_COLUMN_NAME,
    save_as_pdf=True,
    file_name='pairwise_distributions_local_outliers',
    legend_ncol=3,
    hue_order=hue_order_local
    )
    # title='Párové rozložení proměnných s lokálními odlehlými hodnotami podle druhu')
plot_pairwise_distributions(
    df_with_global_outliers,
    species_column=SPECIES_COLUMN_NAME,
    save_as_pdf=True,
    file_name='pairwise_distributions_global_outliers',
    hue_order=hue_order_global,
    legend_ncol=3,
    )
plot_pairwise_distributions(
    df_with_contextual_outliers,
    species_column=SPECIES_COLUMN_NAME,
    markers=["o", "s", "D"],
    save_as_pdf=True,
    file_name='pairwise_distributions_contextual_outliers',
    hue_order=hue_order_contextual,
    )
plot_pairwise_distributions(
    df_with_collective_outliers,
    species_column=SPECIES_COLUMN_NAME,
    save_as_pdf=True,
    file_name='pairwise_distributions_collective_outliers',
    legend_ncol=4,
    )


# df = df.drop(columns='Id')
# # plot_pairwise_distributions(df)

plot_species_boxplots(df, species_column='Druh',
                       save_as_pdf=True, file_name='iris_boxplots')

### PCA
# features = df_with_local_outliers.drop([SPECIES_COLUMN_NAME, 'IsOutlier'], axis=1)
# species = df_with_local_outliers[SPECIES_COLUMN_NAME]
# is_outlier = df_with_local_outliers[IS_OUTLIER_COLUMN_NAME]

# pca = PCA(n_components=2)  # Reduce to 2 dimensions
# principal_components = pca.fit_transform(features)

# pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# pca_df[SPECIES_COLUMN_NAME] = species
# pca_df[IS_OUTLIER_COLUMN_NAME] = is_outlier

# pca_df[SPECIES_COLUMN_NAME] = np.where(pca_df['IsOutlier'], pca_df[SPECIES_COLUMN_NAME] + ' (Outlier)', pca_df[SPECIES_COLUMN_NAME])
# print(pca_df.head())

# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=pca_df, x='PC1', y='PC2', style=SPECIES_COLUMN_NAME, hue=IS_OUTLIER_COLUMN_NAME, palette='bright')
# plt.title('PCA of Iris Dataset')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.show()