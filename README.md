# Alzheimers-Research 

Predicting diagnosis of Alzheimer's Disease using Brain MRI scans 

## Installing Package (Mandatory)
- run `pip3 install -e .` from `Alzheimers-Research/` to install the codebase as a package

### File Structure

- `research/main.py`: Main driver for training and evaluating the model
- `research/datasets/`: Dataset parsers
	- `research/datasets/classification_dataset.py`: main dataset for parsing the classification data
- `research/models/`: Directory for models being evaluated
- `research/util/`: Directory for different helper classes and smaller scripts
