# %%

import pandas as pd
import os

RESULT_DIR = '/home/cbernasconi/et/experiments/cross_dataset_v2/results/wandb_export'

GENERAL_TEMPLATE = '{}/{}/metrics_aggregated.csv'
MACRO_TYPES_TEMPLATE = '{}/{}/macro_types_filtered_ftest_gt2.csv'

K = [5, 10, 20]
DATA = ['bbn', 'figer', 'ontonotes_shimaoka']
DATA_PRETTY = {'bbn' : 'BBN', 'figer': 'FIGER', 'ontonotes': 'OntoNotes'}

# %%
for data in DATA:
    for k in K:
        result_df = pd.read_csv(os.path.join(RESULT_DIR, GENERAL_TEMPLATE.format(data, k)))
        result_df = result_df[result_df['metric'].apply(lambda x : x in ['test/macro_example', 'test/micro'])]

        macro_types_df = pd.read_csv(os.path.join(RESULT_DIR, MACRO_TYPES_TEMPLATE.format(data, k)))

        metrics = ['test/macro_example', 'test/micro']
        projectors = list(set(list(result_df['projector'].values)))

        new_df_rows = []

        for p in projectors:
            row = {'projector' : p.replace('_adapter_bert_ms', '')}
            
            src = p.split('_')[0]
            if src in DATA_PRETTY:
                row['src'] = DATA_PRETTY[src]
                row['projector'] = row['projector'].replace(f'{src}_', '')
                row['projector'] = row['projector'].replace(f'shimaoka_', '')
                row['projector'] = row['projector'].replace(f'vertical_', '')
                row['projector'] = row['projector'].replace(f'_', ' ')
            else:
                row['src'] = '-'
                row['projector'] = '-'

            for m in metrics:
                df_row = result_df[(result_df['metric'] == m) & (result_df['projector'] == p)]

                if 'macro' in m:
                    row['macro_example/p'] = round(df_row['precision/mean'].values[0], 3)
                    row['macro_example/r'] = round(df_row['recall/mean'].values[0], 3)
                    row['macro_example/f1'] = round(df_row['f1/mean'].values[0], 3)

                else:
                    row['micro/p'] = round(df_row['precision/mean'].values[0], 3)
                    row['micro/r'] = round(df_row['recall/mean'].values[0], 3)
                    row['micro/f1'] = round(df_row['f1/mean'].values[0], 3)
            
            df_row = macro_types_df[macro_types_df['projector'] == p]
            row['macro_types/p'] = round(df_row['precision/mean'].values[0], 3)
            row['macro_types/r'] = round(df_row['recall/mean'].values[0], 3)
            row['macro_types/f1'] = round(df_row['f1/mean'].values[0], 3)

            new_df_rows.append(row)

        # create and adjust df
        new_df = pd.DataFrame(new_df_rows)
        new_df = new_df.sort_values(['src', 'projector'])
        new_df = new_df[['src', 'projector',
                         'macro_example/p', 'macro_example/r', 'macro_example/f1',
                         'micro/p', 'micro/r', 'micro/f1',
                         'macro_types/p', 'macro_types/r', 'macro_types/f1',
                         ]]

        # save
        out_path = os.path.join(RESULT_DIR, f'{data}/{k}/composed.csv')
        new_df.to_csv(out_path, index=False)


# %%
