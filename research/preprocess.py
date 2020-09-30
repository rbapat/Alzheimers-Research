import os
import pandas as pd
import subprocess
from scipy.signal import medfilt
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection

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
    except:
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
    except:
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
        volume = equalize_hist(volume, bins_num)

        nib.save(nib.Nifti1Image(volume, affine), dst_path)
    except:
        return 0

    return 1




def main():
    df = pd.read_csv('adni_test_data.csv')

    strip_fails = []
    bias_fails = []
    enhance_fails = []

    idx = 0
    for (root, dirs, files) in os.walk(os.path.join('ADNI', 'Original')):
        for file in files:
            if file[-4:] == '.nii':
                src_path = os.path.join(root, file)
                dst_path = src_path.replace('Original', 'FSL')

                if not os.path.exists(os.path.dirname(dst_path)):
                    os.makedirs(os.path.dirname(dst_path))
                
                print("Starting Element %d..." % idx)

                print("Orienting Element %d..." % idx)
                orient2std(src_path, dst_path)

                print("Registering Element %d..." % idx)
                registration(dst_path, dst_path, '/usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz')

                print("Stripping Element %d..." % idx)
                if strip(dst_path, dst_path) == 0:
                    strip_fails.append(file)
                    print("Stripping failed %s" % file)

                '''
                print("Correcting Bias Field Element %d..." % idx)
                if bias_field_correction(dst_path, dst_path) == 0:
                    bias_fails.append(file)
                    print("Correcting failed %s" % file)
                

                print("Enhancing Scan Element %d..." % idx)
                if enhance(dst_path, dst_path) == 0:
                    enhance_fails.append(file)
                    print("Enhancing failed %s" % file)

                print("Finished Element %d\n" % idx)
                '''

                idx += 1

    print("Strip Fails:")
    print(strip_fails)

    print("Bias Fails:")
    print(bias_fails)

    print("Enhancing Fails:")
    print(enhance_fails)

def dataset_test():
    df = pd.read_csv('adni_test_data.csv')
    freq = [0, 0, 0]
    for (root, dirs, files) in os.walk(os.path.join('ADNI', 'FSL')):
        for file in files:
            print(file)
            if file[-7:] == '.nii.gz':
                start_idx = file.rindex('_') + 2;
                image_id = int(file[start_idx:-7])

                subj = df[df["Image ID"] == image_id]
                cid = ["CN", "AD", "MCI"].index(subj.iloc[0]["Research Group"])
                freq[cid] += 1
        
    print(freq)
                

if __name__ == '__main__':
    dataset_test()
