from research.datasets.base_dataset import BaseParser
import pandas as pd
import numpy as np

class DataParser(BaseParser):
    def __init__(self, data_dim, splits = [0.8, 0.2]):
        BaseParser.__init__(self, data_dim)

        self.extension = '.nii.gz'
        self.ground_truth = self.create_truth_dictionary()

        self.create_dataset(splits, "LongRejects_FSL")

    def create_truth_dictionary(self):
        df = pd.read_csv("LongRejects.csv")
        data_dict = {}

        for index, row in df.iterrows():
            obj = [row["LDELTOTAL"], row["MMSE"], row["CDRSB"], row["mPACCdigit"], row["mPACCtrailsB"]]
            data_dict[int(row["IMAGEUID"])] = obj

        return self.normalize_data_dict(data_dict)

    def normalize_data_dict(self, data_dict):
        data = [v for v in data_dict.values()]
        mean, std = np.mean(data, axis = 0), np.std(data, axis = 0)

        for key in data_dict.keys():
            for i in range(len(data_dict[key])):
                data_dict[key][i] = (data_dict[key][i] - mean[i]) / std[i]

        return data_dict

    def create_ground_truth(self, filename):
        image_id = int(filename[filename.rindex('_') + 2:-len(self.extension)])
        data = self.ground_truth[image_id]

        return data

    def get_num_outputs(self):
        return 5