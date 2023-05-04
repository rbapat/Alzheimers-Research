
class Settings:
    DX_CAP = 2
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 500
    IN_DIMS = (182, 218, 182)
    OUTPUT_DIM = 2

    CROSS_VAL = True
    INNER_SPLIT = 3
    OUTER_SPLIT = 5

    DATASET_PATH = '/media/rohan/ThirdHardDrive/Research/Combined_FSL/scans'
    # DATASET_PATH = '/home/jupyter/Combined_FSL'

    EMBEDDING_PATH = '/media/rohan/ThirdHardDrive/Research/Combined_FSL/embeddings'
    # EMBEDDING_PATH = '/home/jupyter/Embedding'

    SPLITS = [0.8, 0.2]
    LOAD_PATHS = False
    
    CLIN_VARS = ['MMSE', 'CDRSB', 'mPACCtrailsB', 'mPACCdigit', 'APOE4', 'ADAS11', 'ADAS13', 'ADASQ4', 'FAQ', 'RAVLT_forgetting', 'RAVLT_immediate', 'RAVLT_learning', 'TRABSCOR']
    VISIT_DELTA = 6
    NUM_VISITS = 3
