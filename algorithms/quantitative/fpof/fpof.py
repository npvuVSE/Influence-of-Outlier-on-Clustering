import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import warnings

# improve UX
warnings.filterwarnings("ignore")


def encode_columns_dataset(dataset):
    """
    Convert columns of a dataset to one-hot encoded columns.
    """
    te = TransactionEncoder()
    return pd.DataFrame(te.fit(dataset).transform(dataset), columns=te.columns_)


def encode_columns_df(df):
    """
    Convert columns of a DataFrame to one-hot encoded columns.
    """
    dataset = df.values.tolist()

    return encode_columns_dataset(dataset)


def calculate_fpof_values(df, frequent_itemsets):
    """
    Calculates the FPOF values for each transaction in the dataset.
    """
    total_itemsets = len(frequent_itemsets)
    fpof_values = {i: 0 for i in range(len(df))}

    # Loop over each transaction
    for transaction_ind, transaction in df.iterrows():
        # Loop over each frequent itemset
        for _, frequent_item in frequent_itemsets.iterrows():
            # Check if the transaction contains the itemset
            if set(frequent_item['itemsets']).issubset(set(transaction)):
                # Sum the support of the itemset
                fpof_values[transaction_ind] += frequent_item['support']
    
    # Normalize the FPOF values by the total number of frequent itemsets
    for transaction_ind in fpof_values:
        fpof_values[transaction_ind] /= total_itemsets
    
    return pd.Series(fpof_values)


def output_top_n_transactions(df, fpof_values, n):
    """
    Outputs the top-n transactions with the lowest FPOF values.
    """
    # Sort the dictionary by value in ascending order
    sorted_transactions = sorted(fpof_values.items(), key=lambda x: x[1])
    top_n_transactions = []

    # Output the top-n transactions
    for i in range(n):
        transaction_id, fpof_value = sorted_transactions[i]
        # print(f'Transaction ID: {transaction_id}, Transaction: {df.iloc[transaction_id].values}, FPOF Value: {fpof_value}')
        top_n_transactions.append(df.iloc[transaction_id].values)
    
    return top_n_transactions


def calculate_contradictoriness(transaction, itemset, support):
    """
    Calculates the contradictoriness of an itemset with respect to a transaction.
    """
    intersection = set(itemset).intersection(set(transaction))
    return (len(itemset) - len(intersection)) * support

# DEBUG
# def find_top_k_contradict_patterns(top_n_transactions, frequent_itemsets, top_k):
#     """
#     Finds the top-k contradict frequent patterns for each transaction.
#     """
#     top_k_contradict_patterns = {}

#     for transaction_ind, transaction in enumerate(top_n_transactions):
#         # print(f'Processing transaction index {transaction_ind}: {transaction}')
#         contradict_patterns = {}
#         for rule_ind, frequent_item in frequent_itemsets.iterrows():
#             # print(f'Processing frequent item {frequent_item["itemsets"]}')
#             if not set(frequent_item['itemsets']).issubset(set(transaction)):
#                 # print(f'Itemset {frequent_item["itemsets"]} is not a subset of transaction {transaction}')
#                 contradict_score = calculate_contradictoriness(transaction, frequent_item['itemsets'], frequent_item['support'])
#                 # print(f'Contradict score for itemset {frequent_item["itemsets"]} with support {frequent_item["support"]}: {contradict_score}')
#                 contradict_patterns[frozenset(frequent_item['itemsets'])] = contradict_score
#                 # print(f'Contradict patterns: {contradict_patterns}')
#             # else :
#                 # print(f'Itemset {frequent_item["itemsets"]} is a subset of transaction {transaction}')
#         sorted_contradict_patterns = sorted(contradict_patterns.items(), key=lambda x: x[1], reverse=True)[:top_k]
#         print(f'Top-k contradict patterns for transaction index {transaction_ind}: {sorted_contradict_patterns}')
#         top_k_contradict_patterns[transaction_ind] = sorted_contradict_patterns

#     print(f'Top-k contradict patterns: {top_k_contradict_patterns}')
#     return top_k_contradict_patterns


def find_top_k_contradict_patterns(top_n_transactions, frequent_itemsets, top_k):
    """
    Finds the top-k contradict frequent patterns for each transaction.
    """
    top_k_contradict_patterns = {}

    for transaction_ind, transaction in enumerate(top_n_transactions):
        contradict_patterns = {}
        for rule_ind, frequent_item in frequent_itemsets.iterrows():
            if not set(frequent_item['itemsets']).issubset(set(transaction)):
                contradict_score = calculate_contradictoriness(transaction, frequent_item['itemsets'], frequent_item['support'])
                contradict_patterns[tuple(frequent_item['itemsets'])] = contradict_score
        sorted_contradict_patterns = sorted(contradict_patterns.items(), key=lambda x: x[1], reverse=True)[:top_k]
        # print(f'Top-k contradict patterns for transaction index {transaction_ind}: {sorted_contradict_patterns}')
        top_k_contradict_patterns[transaction_ind] = sorted_contradict_patterns

    # print(f'Top-k contradict patterns: {top_k_contradict_patterns}')
    return top_k_contradict_patterns


def FPOF(dataset, min_support, top_n, top_k):
    """
    Main function to calculate the FPOF values for a dataset.
    """

    if type(dataset) == pd.DataFrame:
        df = dataset
        df_discretized = encode_columns_df(df)
    elif type(dataset) == list:
        df_discretized = encode_columns_dataset(dataset)
        df = pd.DataFrame(dataset, columns=[f'item{i}' for i in range(len(dataset[0]))])
    else:
        raise ValueError('The dataset must be a list or a pandas DataFrame.')
    
    frequent_itemsets = apriori(df_discretized, min_support=min_support, use_colnames=True)
    # print(f'Number of frequent itemsets: {len(frequent_itemsets)}\n, 5 first frequent itemsets: {frequent_itemsets[:5]}')
    fpof_values = calculate_fpof_values(df, frequent_itemsets)
    top_n_transactions = output_top_n_transactions(df, fpof_values, top_n)
    top_k_contradict_patterns = find_top_k_contradict_patterns(top_n_transactions, frequent_itemsets, top_k)

    return fpof_values, top_n_transactions, top_k_contradict_patterns
