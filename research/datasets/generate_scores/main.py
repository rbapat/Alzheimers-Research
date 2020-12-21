from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

def normalize_data_dict(data_dict):
    data = [v[0] for v in data_dict.values()]
    mean, std = np.mean(data, axis = 0), np.std(data, axis = 0)


    for key in data_dict.keys():
        for i in range(len(data[0])):
            data_dict[key][0][i] = (data_dict[key][0][i] - mean[i]) / std[i]

def create_truth_dictionary(normalize = True):
    df = pd.read_csv("data.csv") # df = pd.read_csv("LongRejects.csv")
    data_dict = {}

    for index, row in df.iterrows():
        obj = [np.array([row["LDELTOTAL"], row["MMSE"], row["CDRSB"], row["mPACCdigit"], row["mPACCtrailsB"]]), row["DX"]]
        data_dict[int(row["IMAGEUID"])] = obj

    if normalize:
        normalize_data_dict(data_dict)

    return data_dict

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
        c_score.append(scores[key])

        x.append(data[3])
        y.append(data[4])
        z.append(data[0])

        if data_dict[key][1] == "CN":
            color.append('#00FF00')
        elif data_dict[key][1] == "MCI":
            color.append('#FFFF00')
        elif data_dict[key][1] == "Dementia":
            color.append('#FF0000')
        else:
            raise Exception("Unknown DX")

    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        return ax.scatter(x, y, z, c=c_score, marker = '.')
    elif dim == 2:
        return plt.scatter(x, y, c=c_score, marker = '.')

# NAIVE, DEFINITELY CAN BE MORE EFFICIENT
def KNN(K, data_dict):
    scores = {}

    for key in data_dict:
        data, _ = data_dict[key]

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

            tmp.append((np.sum(np.square((iter_data - data))), val))

        tmp.sort(key = lambda x: x[0])
        tmp = tmp[:K]

        score = 0.0
        for (dist, v) in tmp:
            score += v

        scores[key] = score / K

    return scores

def write_scores(scores):

    keys = ["IMAGEUID", "SCORE"]
    with open('scores.csv', mode = 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()

        for key in scores:
            data = {"IMAGEUID": key, "SCORE": scores[key]}
            writer.writerow(data)

def main():
    data_dict = create_truth_dictionary(True)

    scores = KNN(100, data_dict)
    plt.set_cmap('jet')

    frame = plot_data(data_dict, scores)

    write_scores(scores)

    plt.colorbar(frame)
    plt.show()

if __name__ == '__main__':
    main()