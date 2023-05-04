from models import DenseNet
from tqdm import tqdm
import nibabel as nib
import numpy as np
import torch
import os

IN_DIMS = (182, 218, 182)
OUTPUT_DIM = 2

'''
Converts all the MRI scans of the longitudinal patients into feature "embeddings" from the neural network.
Essentially, I take a pretrained neural network and save the result of the final convolutional layer; these are the embeddings.
If the network was pretrained well enough, these embeddings should be a lower dimensional representation of the MRI scan that captures important information for the classification task.
'''

def main():
    model = model = DenseNet(IN_DIMS, OUTPUT_DIM, [6, 12, 32, 24], growth_rate = 24, theta = 0.5, drop_rate = 0.0).cuda()
    #model.load_weights(os.path.join('checkpoints_backup', 'embedding_weights.t7'))
    model.load_weights('weights.t7')
    model.eval()

    # create list of scans to read
    file_list = []
    rootname, newname = '/media/rohan/ThirdHardDrive/Combined_FSL_Old', '/home/rohan/Documents/Alzheimers/embeddings_1000'
    for (root, dirs, files) in os.walk(rootname):
        for file in files:
            if file.endswith('.nii'):
                old_path = os.path.join(root, file)
                new_path = old_path.replace(rootname, newname).replace('.nii', '.npy')

                file_list.append((old_path, new_path))

    for old_path, new_path in tqdm(file_list):
        # load scan and normalize it
        mat = nib.load(old_path).get_fdata()
        mat = (mat - mat.min()) / (mat.max() - mat.min()) # min-max normalization
        mat = torch.Tensor(np.array(mat)).cuda()

        features = model.features(mat).cpu().detach().numpy()

        # save as .npy
        os.makedirs(os.path.dirname(new_path), exist_ok = True)
        np.save(new_path, features)



if __name__ == '__main__':
    main()