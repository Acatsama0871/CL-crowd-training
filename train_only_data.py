# train_only_data.py
# only convert train data to jsonlines


import os
import glob
import pandas as pd
import jsonlines
from tqdm import tqdm
from multiprocessing import Pool


if __name__ == "__main__":
    # get folder names
    model_judgement_path = os.path.join('train_only_data', 'model_judgement')
    folder_names = [cur_file for cur_file in os.listdir(model_judgement_path) if os.path.isdir(os.path.join(model_judgement_path, cur_file))]
    # get file names
    file_names = {}
    for cur_folder in folder_names:
        file_names[cur_folder] = glob.glob(os.path.join(model_judgement_path, cur_folder, '*.csv'))
    # jsonlines job
    def job(cur_folder):
        print(f'{cur_folder} started')
        cur_model_ids = [os.path.basename(cur_path).split('.')[0] for cur_path in file_names[cur_folder]]
        cur_dfs = [pd.read_csv(cur_path) for cur_path in tqdm(file_names[cur_folder])]
        cur_questions_ids = cur_dfs[0]['ID'].to_list()
        records = []
        for i in tqdm(range(len(cur_model_ids))):
            temp_records = {cur_id: cur_dfs[i][cur_dfs[i]['ID'] == cur_id]['Judgement'].to_list()[0] for cur_id in cur_questions_ids}
            records.append({"subject_id": str(i), "responses": temp_records})
        with jsonlines.open(os.path.join(model_judgement_path, cur_folder + '.jsonlines'), 'w') as writer:
            writer.write_all(records)
        print(cur_folder, 'finished')
    # run with multi-processing
    pool = Pool(6)
    pool.map(job, folder_names)
    print("Format Jasonlines Job Finished")
    print('-' * 30)
