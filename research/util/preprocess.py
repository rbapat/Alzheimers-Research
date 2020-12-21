from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
from research.util.util import data_utils
from multiprocessing import Pool
from scipy.signal import medfilt
import multiprocessing as mp
import nibabel as nib
import pandas as pd
import numpy as np
import subprocess
import argparse
import os

'''
TO CONFIGURE FSL ENVIRONMENT:

export ANTSPATH=/opt/ANTs/bin/
export PATH=${ANTSPATH}:$PATH
export FSLDIR=/usr/local/fsl
. ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH
'''


def orient2std(src_path, dst_path):
    command = ["fslreorient2std", src_path, dst_path]
    subprocess.call(command)


def registration(src_path, dst_path, ref_path):
    command = ["flirt", "-in", src_path, "-ref", ref_path, "-out", dst_path,
               "-bins", "256", "-cost", "corratio", "-searchrx", "0", "0",
               "-searchry", "0", "0", "-searchrz", "0", "0", "-dof", "12",
               "-interp", "spline"]
    subprocess.call(command, stdout=open(os.devnull, "r"),
                    stderr=subprocess.STDOUT)

def strip(src_path, dst_path, frac="0.4"):
    try:
        command = ["bet", src_path, dst_path, "-R", "-f", frac, "-g", "0"]
        subprocess.call(command)
    except Exception as e:
        print(e)
        return 0

    return 1

def bias_field_correction(src_path, dst_path):
    try:
        n4 = N4BiasFieldCorrection()
        n4.inputs.input_image = src_path
        n4.inputs.output_image = dst_path

        n4.inputs.dimension = 3
        n4.inputs.n_iterations = [100, 100, 60, 40]
        n4.inputs.shrink_factor = 3
        n4.inputs.convergence_threshold = 1e-4
        n4.inputs.bspline_fitting_distance = 300
        n4.run()
    except Exception as e:
        print(e)
        return 0

    return 1

def rescale_intensity(volume, percentils=[0.5, 99.5], bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    min_value = np.percentile(obj_volume, percentils[0])
    max_value = np.percentile(obj_volume, percentils[1])

    if bins_num == 0:
        obj_volume = (obj_volume - min_value) / (max_value - min_value).astype(np.float32)
    else:
        obj_volume = np.round((obj_volume - min_value) / (max_value - min_value) * (bins_num - 1))
        obj_volume[np.where(obj_volume < 1)] = 1
        obj_volume[np.where(obj_volume > (bins_num - 1))] = bins_num - 1

    volume = volume.astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume

    return volume

def equalize_hist(volume, bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    hist, bins = np.histogram(obj_volume, bins_num, normed=True)
    cdf = hist.cumsum()
    cdf = (bins_num - 1) * cdf / cdf[-1]

    obj_volume = np.round(np.interp(obj_volume, bins[:-1], cdf)).astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume

def enhance(src_path, dst_path):
    try:
        mat = nib.load(src_path)
        
        volume, affine = mat.get_data(), mat.get_affine()
        
        volume = medfilt(volume, 3)
        
        volume = rescale_intensity(volume)
        
        #volume = equalize_hist(volume)
        volume = data_utils.crop_scan(volume)

        nib.save(nib.Nifti1Image(volume, affine), dst_path)
    except Exception as e:
        print(e)
        return 0

    return 1

def process_img(src_path, dst_path):
    if not os.path.exists(os.path.dirname(dst_path)):
        os.makedirs(os.path.dirname(dst_path))
    
    orient2std(src_path, dst_path)
    
    registration(dst_path, dst_path, '/usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz')

    if strip(dst_path, dst_path) == 0:
        print("Stripping failed %s" % src_path)
    
    dst_path += ".gz"

    if bias_field_correction(dst_path, dst_path) == 0:
        print("Correcting failed %s" % src_path)

    if enhance(dst_path, dst_path) == 0:
        print("Enhancing failed %s" % src_path)

    print("Finished", dst_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="directory to preprocess")
    args = parser.parse_args()

    # this is very hacky and bad i know but its a quick fix
    root_dir = os.getcwd()[:os.getcwd().rindex('/')]

    paths = []
    for (root, dirs, files) in os.walk(os.path.join(root_dir, 'ADNI', args.dir)):
        for file in files:
            if file[-4:] == '.nii':
                # also pretty hacky and bad
                p = os.path.join(root, file)
                d = p.replace(args.dir, "%s_FSL" % args.dir)                

                paths.append((p, d))

    with Pool(processes = mp.cpu_count()) as pool:
        pool.starmap(process_img, paths)
                

if __name__ == '__main__':
    main()
