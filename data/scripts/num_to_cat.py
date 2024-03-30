import pandas as pd

def lymphography_convert_to_named_categorical(df):
    mappings = {
        # 'class': {1: 'normal find', 2: 'metastases', 3: 'malign lymph', 4: 'fibrosis'},
        'lymphatics': {1: 'normal', 2: 'arched', 3: 'deformed', 4: 'displaced'},
        # 'block of affere': {1: 'no', 2: 'yes'},
        # 'bl. of lymph. c': {1: 'no', 2: 'yes'},
        # 'bl. of lymph. s': {1: 'no', 2: 'yes'},
        # 'by pass': {1: 'no', 2: 'yes'},
        # 'extravasates': {1: 'no', 2: 'yes'},
        # 'regeneration of': {1: 'no', 2: 'yes'},
        # 'early uptake in': {1: 'no', 2: 'yes'},
        # 'lym.nodes dimin': {1: '0-3'},
        # 'lym.nodes enlar': {1: '1-4'},
        # 'changes in lym': {1: 'bean', 2: 'oval', 3: 'round'},
        # 'defect in node': {1: 'no', 2: 'lacunar', 3: 'lac. marginal', 4: 'lac. central'},
        # 'changes in node': {1: 'no', 2: 'lacunar', 3: 'lac. margin', 4: 'lac. central'},
        # 'changes in stru': {1: 'no', 2: 'grainy', 3: 'drop-like', 4: 'coarse', 5: 'diluted', 6: 'reticular', 7: 'stripped', 8: 'faint'},
        # 'special forms': {1: 'no', 2: 'chalices', 3: 'vesicles'},
        # 'dislocation of': {1: 'no', 2: 'yes'},
        # 'exclusion of no': {1: 'no', 2: 'yes'},
        # 'no. of nodes in': {1: '0-9', 2: '10-19', 3: '20-29', 4: '30-39', 5: '40-49', 6: '50-59', 7: '60-69', 8: '>=70'}
    }

    for x in df:
        df[x] = df[x].map(mappings)

    # df = df.applymap(mappings.get)

    return df
