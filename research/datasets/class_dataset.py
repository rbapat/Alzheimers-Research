from research.datasets.base_dataset import BaseParser
import pandas as pd
import numpy as np
import torch

class DataParser(BaseParser):
    def __init__(self, data_dim, splits = [0.8, 0.2]):
        BaseParser.__init__(self, data_dim)

        self.extension = '.nii.gz'
        self.ground_truth = self.create_truth_dictionary()

        self.create_dataset(splits, "AllData_FSL")
        
    '''
    def create_truth_dictionary(self):
        df = pd.read_csv("scores.csv")
        data_dict = {}
        
        for index, row in df.iterrows():
            if float(row["SCORE"]) > 0.5:
                data_dict[int(row["IMAGEUID"])] = torch.Tensor([0, 1])
            else:
                data_dict[int(row["IMAGEUID"])] = torch.Tensor([1, 0])

            #data_dict[int(row["IMAGEUID"])] = float(row["SCORE"])

        return data_dict
    '''

    def create_truth_dictionary(self):
        df = pd.read_csv("ADNIMERGE.csv", low_memory = False)
        score_df = pd.read_csv("scores.csv")

        data_dict = {}

        for index, row in score_df.iterrows():
            pt = df[df["IMAGEUID"] == row["IMAGEUID"]]
            idx = ["CN", "Dementia", "MCI"].index(pt["DX"].values[0])

            one_hot = np.zeros(self.get_num_outputs())

            if idx > self.get_num_outputs() - 1:
                continue

            one_hot[idx] = 1
            data_dict[int(row["IMAGEUID"])] = one_hot

        return data_dict

    def create_ground_truth(self, filename):
        try:
            image_id = int(filename[filename.rindex('_') + 2:-len(self.extension)])
            data = self.ground_truth[image_id]

            return data

        except Exception:
            return None

    def get_num_outputs(self):
        return 2
