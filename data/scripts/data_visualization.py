import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# from data.scipts.get_data import get_data_numerical, get_data_categorical
# from data.scipts.get_data_from_csv import get_data_from_csv, convert_iris_to_categorical
# from data.scipts.data_transformations import split_df, concat_df

# file_path_raw_data = '/Users/ngocphuong.vu/skola/diplomka/Influence-of-Outlier-on-Clustering/data/numerical/Iris-czech.csv'

# df = get_data_from_csv(file_path_raw_data)

# # df = concat_df(df_raw, df_outliers)
# X, y = split_df(df)

FEATURE_NAMES = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
PLOTS_FILEPATH = 'plots/'

feature_names = {
    'SepalLengthCm': 'Délka kališního lístku [cm]',
    'SepalWidthCm': 'Šířka kališního lístku [cm]',
    'PetalLengthCm': 'Délka okvětního lístku [cm]',
    'PetalWidthCm': 'Šířka okvětního lístku [cm]'
}

feature_names_no_cm = {
    'SepalLengthCm': 'Délka kališního lístku',
    'SepalWidthCm': 'Šířka kališního lístku',
    'PetalLengthCm': 'Délka okvětního lístku',
    'PetalWidthCm': 'Šířka okvětního lístku'
}

feature_names_skloneno = {
    'SepalLengthCm': 'délku kališního lístku',
    'SepalWidthCm': 'šířku kališního lístku',
    'PetalLengthCm': 'délku okvětního lístku',
    'PetalWidthCm': 'šířku okvětního lístku'
}

def plot_species_boxplots(df, title=None, save_as_pdf=False, species_column='Species', file_name=None):
    features = FEATURE_NAMES

    sns.set_theme(style="whitegrid")
    # sns.set_style(rc = {'axes.facecolor': 'lightsteelblue'})
    # sns.set_style("white")
    
    for feature in features:
        f = plt.figure(figsize=(10, 6))
        
        flierprops = dict(marker='o', markerfacecolor='red', markersize=8, linestyle='none', markeredgecolor='gray')
        
        ax = sns.boxplot(x=species_column, y=feature, data=df, palette="muted", flierprops=flierprops, hue=species_column)
        ax.set_xlabel('Druh', fontsize=14)
        ax.set_ylabel(feature_names[feature], fontsize=14)
        
        plt.title(f'Krabicové grafy pro {feature_names_skloneno[feature]} podle druhů', fontsize=16)
        # plt.title(title, fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        if save_as_pdf:
            f.savefig(f'{PLOTS_FILEPATH}{file_name}_{feature}.pdf', bbox_inches='tight')
        else:
            plt.show()
        # f.savefig(f"iris_boxplots_{feature}.pdf", bbox_inches='tight')


def plot_pairwise_distributions(df, title=None, save_as_pdf=False, species_column='Species', file_name=None, markers=["o", "s", "D", 'P'], legend_ncol=3, hue_order=None):
    features = feature_names
    df_renamed = df.rename(columns=features)
    sns.set_theme(style="white")
    
    sns.set_context("notebook", font_scale=1.5)
    g = sns.pairplot(df_renamed, vars=features.values(), hue=species_column, height=3.5, markers=markers, plot_kws={'s': 100, 'alpha': 0.77}, palette='deep', hue_order=hue_order)
    # g = sns.pairplot(df_renamed, vars=features.values(), hue='IsOutlier', height=3.5, markers=["o", "s", "D"], plot_kws={'s': 77})
    if title is not None:
        g.fig.suptitle('Párové rozložení proměnných podle druhu', y=1.03)
        g.fig.suptitle(title, y=1.03)
    sns.move_legend(g, "upper left", bbox_to_anchor=(0, 1.065), ncol=legend_ncol)
    # g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
    
    # Adjust the size of the legend
    # plt.setp(g._legend.get_texts(), fontsize='16')  # for legend text
    # plt.setp(g._legend.get_title(), fontsize='18')  # for legend title
    handles = g._legend_data.values()
    labels = g._legend_data.keys()
    plt.rc('legend', fontsize=25, title_fontsize=25)
    
    # plt.tight_layout()
    if save_as_pdf:
        g.savefig(f'{PLOTS_FILEPATH}{file_name}.pdf', bbox_inches='tight', format='pdf')
    else:
        plt.show()
    # g.savefig(f"iris_pairwise_distribution.pdf", bbox_inches='tight')


# df = df.drop(columns='Id')
# # plot_pairwise_distributions(df)

# plot_species_boxplots(df, species_column='Druh',
#                        save_as_pdf=True, file_name='iris_boxplots')