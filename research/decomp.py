import skimage.transform as transform
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch
import xml.etree.ElementTree as ET
import os



class DataParser:
    def __init__(self, num_output, splits = [0.8, 0.2]):
        self.num_output = num_output
        
        self.create_dataset(splits)

    def extract_cid(self, abs_path):
        root = ET.parse(abs_path).getroot()

        group = root.findall('project/subject/researchGroup')

        if len(group) != 1:
            raise Exception("Number of research group is not 1! %s" % abs_path)

        return ["CN", "AD", "MCI"].index(group[0].text)

    def assemble_xml(self, metadata_path): 
        root = ET.parse(metadata_path).getroot()    

        subject_id = root.findall('subject')[0].get('id')
        series_id = root.findall('series')[0].get('uid')
        image_id = root.findall('image')[0].get('uid')

        return 'ADNI_%s_FreeSurfer_Cross-Sectional_Processing_brainmask_%s_%s.xml' % (subject_id, series_id, image_id)

    def create_dataset(self, splits):
        dxdict = {}
        path = os.path.join('ADNI', 'FreeSurfer')
        for file in os.listdir(path):
            abs_path = os.path.join(path, file)
            if file[-3:] == 'xml' and os.path.isfile(abs_path):
                dxdict[file] = self.extract_cid(abs_path)

        dataset = []
        for (root, dirs, files) in os.walk(path):
            for idx, file in enumerate(files):
                if file[-3:] == 'nii':
                    path = os.path.join(root, file)

                    xml_file = self.assemble_xml(os.path.join(root, files[1 - idx]))
                    cid = dxdict[xml_file]

                    if cid < self.num_output:
                        dataset.append((path, cid))

        random.shuffle(dataset)

        if sum(splits) != 1.0:
            raise Exception("Dataset splits does not sum to 1")

        self.subsets = []
        minIdx, maxIdx = 0, 0

        # split the dataset into the specified chunks
        for idx, split in enumerate(splits):
            chunk = int(len(dataset) * split)
            maxIdx += chunk

            subset = dataset[minIdx:maxIdx]
            random.shuffle(subset)

            self.subsets.append(subset)
            minIdx += chunk

def render_scan(scan):
    fig, axes = plt.subplots(1, len(scan.shape))
    for i in range(scan.shape[1]):

        for j in range(len(scan.shape)):
            axes[j].cla()

        subscans = [
                    scan[min(i, scan.shape[0]) - 1, :, :],
                    scan[:, min(i, scan.shape[1]) - 1, :],
                    scan[:, :, min(i, scan.shape[2] - 1)]
                    ]

        for j in range(len(scan.shape)):
            axes[j].imshow(subscans[j], cmap="gray", origin="lower")

        plt.pause(0.0001)

    plt.show()

def main():
    parser = DataParser(3)
    dataset = parser.subsets[0]

    mat_shape = nib.load(dataset[0][0]).get_fdata().squeeze().shape
    flat_shape = nib.load(dataset[0][0]).get_fdata().flatten().shape

    avg_vec = np.zeros(flat_shape)
    scatter_vec = np.zeros(flat_shape)

    for idx, (path, cid) in enumerate(dataset):
        print("[%d/%d] Average Vector" % (idx + 1, len(dataset)))
        vector = nib.load(path).get_fdata().flatten()
        vector /= len(dataset)

        avg_vec += vector
    
    for idx, (path, cid) in enumerate(dataset):
        print("[%d/%d] Scatter Vector" % (idx + 1, len(dataset)))
        vector = nib.load(path).get_fdata().flatten()
        scatter_vec += ((vector - avg_vec) * (vector - avg_vec).T) / len(dataset)

    scatter_mat = np.reshape(scatter_vec, mat_shape)
    avg_mat = np.reshape(avg_vec, mat_shape)

    eigen_val, eigen_vec = np.linalg.eig(scatter_mat)

    
    eig_pairs = [(eigen_val[index], eigen_vec[:,index]) for index in range(len(eigen_val))]

    pair = eig_pairs[0]
    rside = pair[0] * pair[1]

    lside = nib.load(dataset[0][0]).get_fdata() * pair[1]

    print(rside.shape, lside.shape)

        

    


if __name__ == '__main__':
    main()
