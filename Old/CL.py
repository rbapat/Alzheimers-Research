import numpy as np
import dataset
import random
import torch

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor


# trains model on simple classification task, I use this for pretraining models for the longitudinal task
def main():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    parser = dataset.DataParser(5, 1)

    data_mat = np.zeros((len(parser.loaders[0]), 4))
    colors, dx = [], []
    for idx, (path, clin_vars, ground_truth) in enumerate(parser.loaders[0]):
        path, clin_vars, ground_truth = path[0], clin_vars[0], ground_truth[0]

        image_id = int(path[path.rindex('I')+1:-4])
        data_mat[idx, :] = clin_vars

        colors.append(['g', 'r', 'y'][ground_truth[0]])
        dx.append(ground_truth[0])

    colors = np.array(colors)
    cn, mci, ad = colors == 'g', colors == 'y', colors == 'r'

    pca = PCA(n_components = 2)
    transformed_data = pca.fit_transform(data_mat)

    knn = KNeighborsRegressor(n_neighbors = 100)
    knn.fit(data_mat, dx)

    scored_data = knn.predict(data_mat)
    scored_data = np.abs(scored_data - dx)
    scored_data[scored_data > 1] = 1
    
    plt.title("PCA of dataset of patients (Diagnosis)")
    plt.scatter(transformed_data[:, 1], transformed_data[:, 0], c = colors, marker = '.')
    plt.figure()

    plt.title("PCA of dataset of patients (Difficulty Score)")
    plt.set_cmap('jet')
    frame = plt.scatter(transformed_data[:, 1], transformed_data[:, 0], c = scored_data, marker = '.')
    plt.colorbar(frame)

    plt.show()




if __name__ == '__main__':
    main()