import pandas as pd
import numpy as np
import os

# create mapping from image id to path of scan on disk using dataset path
def get_volume_paths(dataset_path):
    volume_dict = {}

    for (root, dirs, files) in os.walk(dataset_path):
        for file in files:
            if file.endswith('.nii'):
                path = os.path.join(root, file)
                image_id = int(file[file.rindex('I')+1:-4])

                volume_dict[image_id] = path

    return volume_dict

# reader ADNIMERGE.csv, and omit patients that dont have the specified clinical variables. Also omit patients whose scans I haven't downloaded
def get_df(dataset_path, clin_vars):
    volume_dict = get_volume_paths(dataset_path)

    subs = ['PTID', 'Month', 'IMAGEUID', 'DX', 'DX_bl', *clin_vars]
    df = pd.read_csv('ADNIMERGE.csv', low_memory = False).dropna(subset = subs)[subs]
    df = df[df['IMAGEUID'].isin(volume_dict.keys())]

    new_dict = {}
    for image_id in df['IMAGEUID'].values:
        new_dict[int(image_id)] = volume_dict[int(image_id)]

    return new_dict, df

# given a patient's dataframe in ADNIMERGE, return a list of all the sequential visits they've had.
# Essentially, each list sequential visit should have `seq_length` scans that are all `freq` months apart.
# assumes patient dataframe is sorted by month, which I do before calling
def get_num_seq_rows(df, freq, seq_length):
    if len(df) < seq_length:
        return []

    seqs = []
    for i in range(len(df) - seq_length + 1):
        seq_df = df.iloc[i:i+seq_length]
        dxs, months, ids = seq_df["DX"].values, seq_df["Month"].values, seq_df["IMAGEUID"].values

        # make sure each scan is `freq` months apart
        gap_arr = np.unique(np.array(months[:-1]) - np.array(months[1:]))
        if len(gap_arr) == 1 and np.unique(gap_arr)[0] == freq * -1:
            seqs.append((dxs, months, ids))

    return seqs

# dataset_path is the path to all the downloaded .nii scans, clin_vars is a list of clinical variables get from ADNIMERGE
# returns:
#   dictionary mapping IMAGEUID to .nii path
#   pandas dataframe of ADNIMERGE.csv, with extra patients filtered out
#   dictionary mapping a patient id (PTID) to the respective class (converter, non-converter, or normal classification if it doesn't have enough timepoints for either)
def create_class_dict(dataset_path, clin_vars):
    paths, df = get_df(dataset_path, clin_vars)
    unique_pt = df['PTID'].unique()

    type_dict = {}
    for pt in unique_pt:
        # get patient dataframe, and all the sequential visits
        pt_df = df[df["PTID"] == pt].sort_values(by = ['Month'])
        seqs = get_num_seq_rows(pt_df, 6, 3)
        num_dx = len(pt_df['DX'].unique())  

        # default to using the patient for classification
        res = (f'CLASSIFICATION_{"_".join(pt_df["DX"].unique())}', None)

        # criteria for non-converters
        if num_dx == 1:
            if 'MCI' in pt_df['DX'].values and pt_df['Month'].values[-1] >= 36 and len(seqs) > 1:
                res = ('NON_CONVERTER', seqs[0][2])
        elif len(seqs) != 0:
            for dxs, months, ids in seqs:
                target = months[0] + 36
                only_mci = 'Dementia' not in dxs #len(np.unique(dxs)) == 1 and dxs[0] == 'MCI'

                final_df = pt_df[pt_df['Month'] >= target]
                final_dx = final_df["DX"].values
                final_has_dem = 'Dementia' in final_dx

                if only_mci and final_has_dem: # if patient doesn't have dementia in the 3 timepoints, but has dementia 3 years later: use as converter
                    res = ('CONVERTER', ids)
                    break
                elif only_mci and not final_has_dem: # if patient doesn't have dementia in the 3 timepoints or 3 years later: use as non-converter
                    res = ('NON_CONVERTER', ids)
                    break

        
        type_dict[pt] = res

    return paths, df, type_dict

# print out number of patients found
def main():
    paths, df, type_dict = create_class_dict('/media/rohan/ThirdHardDrive/Combined_FSL', ['MMSE', 'CDRSB', 'mPACCtrailsB', 'mPACCdigit'])

    classification, converters, non_converters = [], 0, 0
    for pt_key in type_dict:
        val, found = type_dict[pt_key]

        if val == 'NON_CONVERTER':
            non_converters += 1
            print(found)
        elif val == 'CONVERTER':
            converters += 1
        elif val.startswith('CLASSIFICATION_'):
            classification.append(val[len('CLASSIFICATION_'):].split('_'))

    print(len(classification), converters, non_converters)


    

if __name__ == '__main__':
    main()