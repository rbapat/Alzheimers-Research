from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import csv

def get_class_sets(df):
    cn_mci, mci_ad, cn_cn, mci_mci, ad_ad = set(), set(), set(), set(), set()
    for scan in df.iloc:
        if scan["DX_bl"] == "CN" and scan["DX"] == "MCI":
            cn_mci.add(scan["PTID"])
        elif (scan["DX_bl"] == "LMCI" or scan["DX_bl"] == "EMCI") and scan["DX"] == "Dementia":
            mci_ad.add(scan["PTID"])

        d = np.unique(df[df["PTID"] == scan["PTID"]]["DX"].values)

        if len(d) == 1 and d[0] == "CN":
            cn_cn.add(scan["PTID"])
        elif len(d) == 1 and d[0] == "MCI":
            mci_mci.add(scan["PTID"])
        elif len(d) == 1 and d[0] == "Dementia":
            ad_ad.add(scan["PTID"])

    return cn_mci, mci_ad, cn_cn, mci_mci, ad_ad

def concat_df(df, targets, dxkeys):
    frames = []
    for target, key in zip(targets, dxkeys):
        dx_df = df[df["PTID"].isin(target)]
        dx_bl = dx_df[dx_df["Month"].isin([0])]
        #dx_bl["DXKEY"] = [key] * len(dx_bl)
        frames.append(dx_bl)

    return pd.concat(frames)

def create_truth_dictionary(df, variables):
    data_dict = {}

    for index, row in df.iterrows():
        # array is [[features], dx, dxkey]
        #obj = [np.array(row[variables].astype(float)), row["DX"], row["DXKEY"]]
        obj = [np.array(row[variables].astype(float)), row["DX"]]
        data_dict[int(row["IMAGEUID"])] = obj


    # normalize
    data = [v[0] for v in data_dict.values()]
    mean, std = np.mean(data, axis = 0), np.std(data, axis = 0)

    for key in data_dict.keys():
        for i in range(len(data[0])):
            data_dict[key][0][i] = (data_dict[key][0][i] - mean[i]) / std[i]

    return data_dict

# NAIVE, DEFINITELY CAN BE MORE EFFICIENT
def KNN(K, data_dict):
    scores = {}

    for key in data_dict:
        data, new_dx = data_dict[key]

        tmp = []
        for iter_key in data_dict:

            '''
            # TODO: dont include current key (?)
            if iter_key == key:
                continue
            '''

            iter_data, iter_dx = data_dict[iter_key]

            if iter_dx == "CN":
                val = 0.0
            elif iter_dx == "MCI":
                val = 0.5
            elif iter_dx == "Dementia":
                val = 1.0
            else:
                raise Exception("Unknown DX")

            tmp.append((np.linalg.norm(iter_data - data), val))

        tmp.sort(key = lambda x: x[0])
        tmp = tmp[:K]

        score = 0.0
        for (dist, v) in tmp:
            score += v

        scores[key] = score / K

    return scores

def plot_data(data_dict, scores, dim = 3):
    x_l = []
    y_l = []

    x = []
    y = []
    z = []
    c_score = []
    color = []

    for key in data_dict:
        data = data_dict[key][0]

        try:
            c_score.append(scores[key])
        except Exception:
            continue

        x.append(data[3])
        y.append(data[4])
        z.append(data[0])

        '''
        ax_.set_xlabel('mPACCDigit')
        ax_.set_ylabel('mPACCtrailsB')
        ax_.set_zlabel('LDELTOTAL')
        '''

        if data_dict[key][1] == "CN":
            color.append('#00FF00') # green
        elif data_dict[key][1] == "MCI":
            color.append('#FFFF00') # yellow
        elif data_dict[key][1] == "Dementia":
            color.append('#FF0000') # red
        else:
            raise Exception("Unknown DX")

    if dim == 3:
        ax_ = plt.axes(projection = '3d')
        ax_.scatter(x,y,z, c = color, marker = '.')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        return ax.scatter(x, y, z, c=c_score, marker = '.')
    elif dim == 2:
        plt.scatter(x, y, c=color, marker = '.')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        return ax.scatter(x, y, c=c_score, marker = '.')

def write_scores(scores, filename):

    keys = ["IMAGEUID", "SCORE"]
    with open(filename, mode = 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()

        for key in scores:
            data = {"IMAGEUID": key, "SCORE": scores[key]}
            #print(data, len(scores))
            writer.writerow(data)

def main():
    keys = ["VISCODE", "PTID", "IMAGEUID", "LDELTOTAL", "MMSE", "CDRSB", "mPACCdigit", "mPACCtrailsB", "DX", "AGE"]
    df = pd.read_csv('ADNIMERGE.csv', low_memory = False).dropna(subset = keys) 

    cn_mci, mci_ad, cn_cn, mci_mci, ad_ad = get_class_sets(df)

    #df = df[keys]
    #targets = [cn_cn, ad_ad, mci_mci, cn_mci, mci_ad]
    #dxkeys = ["CNCN", "ADAD", "MCIMCI", "CN_MCI", "MCI_AD"]

    targets = [cn_cn, ad_ad, mci_mci]
    dxkeys = ["CNCN", "ADAD", "MCIMCI"]

    score_df = concat_df(df, targets, dxkeys)
    truth_df = create_truth_dictionary(score_df, ["CDRSB", "LDELTOTAL", "mPACCdigit", "mPACCtrailsB", "MMSE", "AGE"])

    scores = KNN(300, truth_df)
    write_scores(scores, "scores.csv")

    '''
    plt.set_cmap('jet')

    frame = plot_data(truth_df, scores)

    plt.colorbar(frame)
    plt.show()
    '''
    


if __name__ == '__main__':
    main()
