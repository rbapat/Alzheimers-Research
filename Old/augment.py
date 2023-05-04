import scipy.ndimage.interpolation as interp
import multiprocessing as mp
import nibabel as nib
import numpy as np
import dataset
import random
import torch
import os

def generate_noise(mat, max_intensity, ratio = 0.025):
    total_points = int(mat.size * ratio)
    dims = mat.shape

    coords = np.random.rand(total_points, 3)
    coords[:, 0] *= dims[0]
    coords[:, 1] *= dims[1]
    coords[:, 2] *= dims[2]

    out = np.copy(mat)
    for x, y, z in coords.astype(int):
        out[x, y, z] = random.randint(0, 1) * max_intensity

    return out

def augment(path):
    img = nib.load(path)
    mat = img.get_fdata()

    rotated_mat = interp.rotate(mat, angle = random.randint(-12, 12))
    flipped_mat = np.flip(mat, 0)
    noisy_mat = generate_noise(mat, np.max(mat))

    names = ['rotated', 'flipped', 'noisy']
    mats = [rotated_mat, flipped_mat, noisy_mat]

    for name, matrix in zip(names, mats):
        new_img = nib.Nifti1Image(mat, img.affine)
        nib.save(new_img, path.replace('.nii', f'_{name}.nii'))

def main():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    parser = dataset.DataParser(3, 1)
    aug_paths = []
    for loader in parser.loaders:
        for vol_paths, clin_var, dx in loader:
            for path in vol_paths:
                aug_paths.append(path[0])

    print(f"Processing {len(aug_paths)} files")
    with mp.Pool(processes = mp.cpu_count()) as pool:
        pool.map(augment, aug_paths)


if __name__ == '__main__':
    main()