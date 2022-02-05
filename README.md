# Alzheimers-Research 

Predicting conversion from MCI (mild cognitive impairment) to Dementia using multiple timepoints and eventually both imaging and non-imaging data.

I haven't included any of the actual MRI scans here, this is just the code I'm using.

### Relevant Files

- `classification.py` trains the neural network on the basic AD/NC classification task
- `longitudinal.py` trains the network on the longitudinal conversion tasks, it's still a work in progress
- `models.py` contains the implementation of the neural network model I'm using (DenseNet), and the LSTM
- `dataset.py` handles all the the data parsing before feeding it into the neural network, and it uses `dataset_helper.py` internally to generate the list of patients to use
- `preprocess.py` takes in a directory and preprocesses (skull strips and registers) all of the .nii scans found
- `ADNIMERGE.csv` is the non-imaging data downloaded from ADNI, `MNI152_T1_1mm.nii.gz` is the standard 1mm template that I'm registering to
