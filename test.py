from torch.utils.data import DataLoader
from util import data_utils as du
from dataset import DataParser
from extractor import Extractor
import matplotlib.pyplot as plt
import multiprocessing
import nibabel as nib

DATA_DIM = (128, 128, 64)
NUM_OUTPUTS = 2

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
    '''
    dataset = DataParser("dataset.csv", DATA_DIM, NUM_OUTPUTS, splits = [1.0])

    loader = DataLoader(dataset.get_loader(0), batch_size = 1, shuffle = True, num_workers = multiprocessing.cpu_count())

    for data, label in loader:
        scan = data[0]

        du.render_scan(scan)
        input()
    '''
    path = "C:\\Users\\Rohan\\Documents\\Code\\Alzheimers-Research\\Original\\MCI\\002_S_0782\\MPR__GradWarp__B1_Correction__N3\\2006-08-14_09_39_47.0\\S17835\\ADNI_002_S_0782_MR_MPR__GradWarp__B1_Correction__N3_Br_20070217003330133_S17835_I40716.nii"

    mat = nib.load(path)

    extractor = Extractor()
    prob = extractor.run(mat.get_fdata())
    mask = prob > 0.5

    brain = mat.get_fdata()[:]
    brain[~mask] = 0
    brain = du.crop_scan(brain)

    
    render_scan(brain)
    



if __name__ == '__main__':
    main()