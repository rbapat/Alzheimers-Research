from dataset_helper import create_class_dict
import numpy as np
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch

DX_CAP = 2
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
NUM_EPOCHS = 500
IN_DIMS = (182, 218, 182)
OUTPUT_DIM = 2

CROSS_VAL = True
INNER_SPLIT = 3
OUTER_SPLIT = 5

DATASET_PATH = '/media/rohan/ThirdHardDrive/Combined_FSL'
# DATASET_PATH = '/home/jupyter/Combined_FSL'

EMBEDDING_PATH = '/home/rohan/Documents/Alzheimers/embeddings_288'
# EMBEDDING_PATH = '/home/jupyter/Embedding'

SPLITS = [0.8, 0.2]
LOAD_PATHS = False

# ['PTID', 'IMAGEUID', 'DX', 'DX_bl', 'Month']
CLIN_VARS = ["AGE", "PTGENDER"]#['MMSE', 'CDRSB', 'mPACCtrailsB', 'mPACCdigit', 'APOE4', 'ADAS11', 'ADAS13', 'ADASQ4', 'FAQ', 'RAVLT_forgetting', 'RAVLT_immediate', 'RAVLT_learning', 'TRABSCOR']
VISIT_DELTA = 6
NUM_VISITS = 3

def create_single_tp_loader(df, type_dict, paths, clin_vars):
    gts = ['CN', 'Dementia', 'MCI']
    selected, data = [], []
    
    counter = [0, 0, 0]
    for pt_key in type_dict:
        val, _ = type_dict[pt_key]
        if not val.startswith('CLASSIFICATION_'):
            continue

        pt_df = df[df["PTID"] == pt_key]
        candidates = [pt_df[pt_df["DX"] == dx] for dx in ['Dementia', 'MCI', 'CN']]
        
        # select one of the timepoints this patient has; prioritize dementia, then MCI, then normal diagnosis
        for cand in candidates:
            if len(cand) > 0:
                selected.append(cand[["DX", "IMAGEUID"]].values[0])
                counter[gts.index(selected[-1][0])] += 1
                data.append(cand[clin_vars].values[0])
                break

    data = np.array(data)
    # mean, std = np.mean(data, axis = 0), np.std(data, axis = 0)
    # data = np.apply_along_axis(lambda row: (row - mean) / std, 1, data)

    # A little hacky, but makes sure each group has the same number of patients in the created dataset (omits extra patients)
    dx_cap = DX_CAP
    counter = [min(counter[:dx_cap])] * 3
    
    # Create the actual dataset, omitting patients with MCI if dx_cap == 2 and making sure there are the same number of patients with AD/NC/MCI
    result = []
    for (dx, image_id), data_row in zip(selected, data):
        dx_idx = gts.index(dx)

        if dx_idx < dx_cap and counter[dx_idx] > 0:
            result.append((paths[int(image_id)], data_row, dx_idx))
            counter[dx_idx] -= 1

    return result


def build_longitudinal_dataset( df, type_dict, paths, clin_vars):
    selected, data = [], []
    seq_len = 3
    for pt_key in type_dict:
        val, ids = type_dict[pt_key] # ids is the IMAGEUID field of each timepoint of this patient
        pt_df = df[df["PTID"] == pt_key]

        if val.startswith('CLASSIFICATION_'):
            continue

        # get the rows in ADNIMERGE of this patient's 3 timepoints in order to get clinical variables
        rows = pt_df[pt_df["IMAGEUID"].isin(ids)].sort_values(by = ["Month"])
        dx = ['NON_CONVERTER', 'CONVERTER'].index(val)

        selected.append((rows["IMAGEUID"].values, dx))
        for v in rows[clin_vars].values:
            data.append(v)

    # normalize each clinical variable to have zero mean and unit variance
    data = np.array(data)
    # mean, std = np.mean(data, axis = 0), np.std(data, axis = 0)
    # data = np.apply_along_axis(lambda row: (row - mean) / std, 1, data)

    # Create the actual dataset
    dataset = []
    for idx, (imids, dx) in enumerate(selected):
        volume_paths = [paths[int(image_id)] for image_id in imids]
        dataset.append((volume_paths, data[seq_len*idx:seq_len*idx+seq_len], dx))

    return dataset

def classification():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    paths, df, type_dict = create_class_dict(DATASET_PATH, CLIN_VARS, VISIT_DELTA, NUM_VISITS)
    dataset = create_single_tp_loader(df, type_dict, paths, CLIN_VARS)
    all_idx = list(range(len(dataset)))

    dx_list = np.array([dataset[i][2] for i in all_idx])

    cn = np.argwhere(dx_list == 0).squeeze()
    ad = np.argwhere(dx_list == 1).squeeze()
    types = [('CN', cn), ('AD', ad)]
    
    for name, lst in types:
        print(name)
        ages = np.array([dataset[i][1][0] for i in lst])
        genders = [dataset[i][1][1] for i in lst]

        print(f'Mean Age: {np.mean(ages)}')
        print(f'Std Age: {np.std(ages)}')
        print(f'Num Male: {genders.count("Male")}')
        print(f'Num Female: {genders.count("Female")}')
        print()

def prediction():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    paths, df, type_dict = create_class_dict(DATASET_PATH, CLIN_VARS, VISIT_DELTA, NUM_VISITS)
    dataset = build_longitudinal_dataset(df, type_dict, paths, CLIN_VARS)
    all_idx = list(range(len(dataset)))

    dx_list = np.array([dataset[i][2] for i in all_idx])
    smci = np.argwhere(dx_list == 0).squeeze()
    pmci = np.argwhere(dx_list == 1).squeeze()
    types = [('sMCI', smci), ('pMCI', pmci)]
    

    for name, lst in types:
        print(name)
        ages = np.array([dataset[i][1][0][0] for i in lst])
        genders = [dataset[i][1][0][1] for i in lst]

        print(f'Mean Age: {np.mean(ages)}')
        print(f'Std Age: {np.std(ages)}')
        print(f'Num Male: {genders.count("Male")}')
        print(f'Num Female: {genders.count("Female")}')
        print()

if __name__ == '__main__':
    classification()