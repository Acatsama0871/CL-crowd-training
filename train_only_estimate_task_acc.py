# train_only_estimate_task_acc.py
# estimate the task accuracy

import os
import pickle
import json
import pandas as pd
import numpy as np
from p_tqdm import p_map

def one_task_acc(the_task_id):
    judgements = []
    for i in range(108):  # FIXME: hardcoded number of models
        cur_df = model_judgement_dfs[i]
        cur_judgement = cur_df[cur_df['ID'] == the_task_id]['Judgement'].iloc[0]
        judgements.append(cur_judgement)

    return np.sum(judgements) / len(judgements)

if __name__ == '__main__':
    cwd = os.getcwd()
    # colnames
    col_names = ['Study Period', 'Perspective', 'Population', 'Sample Size', 'Intervention', 'Country']
    # estimate accuracy
    all_classes_acc_result = {}
    for cur_col in col_names:
        print(f'{cur_col} started')
        # task difficulties
        fitted_param_path = os.path.join(cwd, 'train_only_data', 'fitted_IRT', cur_col, 'best_parameters.json')
        best_param_json = json.load(open(fitted_param_path))
        ids = [best_param_json['item_ids'][i] for i in best_param_json['item_ids']]
        diff_df = pd.DataFrame({'ID': ids, 'diff': best_param_json['diff']})
        # model judgements
        model_judgement_dfs = [pd.read_csv(os.path.join(cwd, 'train_only_data', 'model_judgement', cur_col, str(i) + '.pth.csv')) for i in range(108)]  # FIXME: hardcoded number of models
        task_IDs = diff_df['ID'].tolist()
        # calculate acc
        acc_result = p_map(one_task_acc, task_IDs, num_cpus=10)
        acc_result_df = pd.DataFrame({'ID': task_IDs, 'acc': acc_result})
        # append
        all_classes_acc_result[cur_col] = pd.merge(diff_df, acc_result_df, on='ID')

    with open(os.path.join(cwd, 'train_only_data', 'all_classes_acc_result.pkl'), 'wb') as f:
        pickle.dump(all_classes_acc_result, f)
