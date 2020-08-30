from pprint import pprint
import nibabel as nib
import os
from research.util.util import data_utils


# quick script to convert all the .mgz files created by freesurfer into .nii, crop them, and put them in the right place.
def main():
    for (root, dirs, files) in os.walk('/home/rohan/Research/ADNI/Original/'):
        for file in files:
            if file =='brainmask.mgz':
                try:
                    target_path = os.path.join(root.replace('Original', 'Processed'), file.replace('mgz', 'nii'))
                    source_path = os.path.join(root, file)

                    if not os.path.exists(os.path.dirname(target_path)):
                        os.makedirs(os.path.dirname(target_path))

                    os.system("mri_convert %s %s" % (source_path, target_path))

                    mat = nib.load(target_path)
                    cropped = data_utils.crop_scan(mat.get_fdata())

                    cropped_nib = nib.Nifti1Image(cropped, mat.affine)
                    nib.save(cropped_nib, target_path)
                    
                except Exception:
                    print("Not Found:", root)


if __name__ == '__main__':
    main()