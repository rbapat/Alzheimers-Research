# Alzheimers-Research 

Predicting diagnosis of Alzheimer's Disease using Brain MRI scans 



### File Structure

- `main.py`: Main driver for training and evaluating the model
- `dataset.py`: Classes for managing and parsing the dataset
- `model.py`: Defines the PyTorch model being used
- `util.py`: Helper classes and functions for the project

### Omissions

- Some files are omitted because it contains ADNI specific data; this includes 
  - `data_query.pdf`: Exact query used to download ADNI data
  - `dataset.csv`: Data labels downloaded from ADNI using custom query
  - `*.t7`: weights trained and outputted by model
  - `Processed/*`: Preprocessed data
  - `Original/*`: Raw data from ADNI