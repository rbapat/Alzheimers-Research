from research.datasets.base_dataset import BaseParser
import pandas as pd
import numpy as np

class DataParser(BaseParser):
    def __init__(self, data_dim, splits = [0.8, 0.2]):
        BaseParser.__init__(self, data_dim)

        self.extension = '.nii.gz'
        self.ground_truth = self.create_truth_dictionary()

        self.create_dataset(splits, "AllData_FSL")
        
    # create mapping from image id to image score
    def create_truth_dictionary(self):
        df = pd.read_csv("scores.csv")
        data_dict = {}

        for index, row in df.iterrows():
            data_dict[int(row["IMAGEUID"])] = float(row["SCORE"])

        return data_dict

    # get image score from the path of a file
    def create_ground_truth(self, filename):
        try:
            image_id = int(filename[filename.rindex('_') + 2:-len(self.extension)])
            data = self.ground_truth[image_id]

            return data
        except Exception: # probably KeyError
            return None

    def get_num_outputs(self):
        return 1
