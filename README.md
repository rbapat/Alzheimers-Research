# Alzheimers-Research 

Predicting diagnosis of Alzheimer's Disease using Brain MRI scans 

## Installing Package (Mandatory)
- run `pip3 install -e .` from `Alzheimers-Research/` to install the codebase as a package

### File Structure

- `research/regr_kfold.py`,`research/val_kfold.py` : Main driver for training and evaluating the current model. `regr_kfold` will train the given model and `val_kfold` will print out the predictions, labels, and classes from the test dataset
- `research/generate_scores.py`: Uses `ADNIMERGE.csv` to generate all of the scores in the dataset with the weighted KNN algorithm
- `research/regr_cam.py`: Generates the class activation mapping of the specified network on an input instance for the regression task
- `research/other_runtimes/`: The main code I used to evaluate other types of models, backed up in case I need it
- `research/datasets/`: Dataset parsers
	- `research/datasets/base_dataset.py`: abstract class that other datasets inherit, implements data splitting
	- `research/datasets/scored_dataset.py`: main dataset for parsing the regression data (scores)
	- `research/datasets/class_dataset.py`: main dataset for parsing the classification data
- `research/models/`: Directory for models being evaluated
  - `research/models/densenet`: DenseNet implementation being used
- `research/util/`: Directory for different helper classes and smaller scripts
  - `research/util/graph_data`: Graphs the correlation and errors of lists of predicted values with labels
  - `research/util/Grapher`: old class used to graph some data while  training
  - `research/util/preprocess`: script to spawn multiple processes and use FSL to preprocess a given datase
