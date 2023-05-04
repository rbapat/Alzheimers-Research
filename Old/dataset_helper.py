import pandas as pd
import numpy as np
import os

# create mapping from image id to path of scan on disk using dataset path
def get_volume_paths(dataset_path):
    volume_dict = {}

    for (root, dirs, files) in os.walk(dataset_path):
        for file in files:
            if 'rotated' in file or 'flipped' in file or 'noisy' in file:
                continue

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
def create_class_dict(dataset_path, clin_vars, seq_separation, seq_length):
    paths, df = get_df(dataset_path, clin_vars)
    unique_pt = df['PTID'].unique()

    type_dict = {}
    for pt in unique_pt:
        # get patient dataframe, and all the sequential visits
        pt_df = df[df["PTID"] == pt].sort_values(by = ['Month'])
        seqs = get_num_seq_rows(pt_df, seq_separation, seq_length)
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

                no_dementia = 'Dementia' not in dxs
                only_mci = len(np.unique(dxs)) == 1 and dxs[0] == 'MCI'


                final_df = pt_df[pt_df['Month'] >= target]
                if len(final_df) == 0:
                    continue
                
                final_dx = final_df["DX"].values
                dementia_after_3y = 'Dementia' in final_dx

                # is_converter = no_dementia and dementia_after_3y
                # is_nonconverter = no_dementia and not dementia_after_3y

                is_converter = dxs[-1] == 'MCI' and dementia_after_3y
                is_nonconverter = dxs[-1] == 'MCI' and not dementia_after_3y

                if is_converter: # if patient doesn't have dementia in the 3 timepoints, but has dementia 3 years later: use as converter
                    res = ('CONVERTER', ids)
                    break
                elif is_nonconverter: # if patient doesn't have dementia in the 3 timepoints or 3 years later: use as non-converter
                    res = ('NON_CONVERTER', ids)
                    break

        
        type_dict[pt] = res

    return paths, df, type_dict

# print out number of patients found
def main():
    paths, df, type_dict = create_class_dict('/media/rohan/ThirdHardDrive/Combined_FSL', ['MMSE', 'CDRSB', 'mPACCtrailsB', 'mPACCdigit', 'PTGENDER', 'AGE'], 6, 3)

    classification, converters, non_converters = [], 0, 0

    #gender = [[0, 0], [0, 0]]
    #ages = [0, 0]

    for pt_key in type_dict:
        val, found = type_dict[pt_key]

        if val == 'NON_CONVERTER':
            #pt = df[df["IMAGEUID"] == found[0]]
            #gender[0][["Male", "Female"].index(pt['PTGENDER'].values[0])] += 1
            #ages[0] += pt["AGE"].values[0]
            non_converters += 1
        elif val == 'CONVERTER':
            #pt = df[df["IMAGEUID"] == found[0]]
            #gender[1][["Male", "Female"].index(pt['PTGENDER'].values[0])] += 1
            #ages[1] += pt["AGE"].values[0]
            converters += 1
        elif val.startswith('CLASSIFICATION_'):
            classification.append(val[len('CLASSIFICATION_'):].split('_'))

    #print(len(classification), converters, non_converters)

    splits = [0, 0, 0]
    for lst in classification:
        splits[['CN', 'MCI', 'Dementia'].index(lst[0])] += 1

    print(splits)


if __name__ == '__main__':
    main()