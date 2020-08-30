from torch.utils.data import DataLoader
import skimage.transform
import skimage.color
from scipy.ndimage import zoom
import torch.optim as optim
import multiprocessing
import copy
import argparse
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
import cv2

from research.models.Inception import EModel
from research.datasets.classification_dataset import DataParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# model hyperparameters
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1000
BATCH_SIZE = 1
WEIGHT_DECAY = 0.00001

# weight/graphing parameters
LOAD_WEIGHT = True
SAVE_FREQ = 10
SAVE_MODEL = True
GRAPH_FREQ = 10
GRAPH_METRICS = True
EARLY_STOP_THRESH = 90

# data shapes
NONIMAGING_FEATURES = 5
DATA_DIM = (128, 128, 64)
NUM_OUTPUTS = 2

'''
*************************************

    I need to clean this up a lot    

*************************************
'''

def get_cam(conv_features, linear_features, idx, small_dims, large_dims):
    cam = linear_features[idx[0].item()].dot(conv_features)
   

    cam = cam.reshape(*small_dims)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)

    full_size = skimage.transform.resize(cam,  large_dims, anti_aliasing = True, preserve_range = True, mode = 'edge')
    return full_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", help="which class (0 = CN, 1 = AD)")
    args = parser.parse_args()

    dataset = DataParser(DATA_DIM, NUM_OUTPUTS, splits = [1.0])
    loader = DataLoader(dataset.get_loader(0), batch_size = 1, shuffle = True)
    loader = iter(loader)

    # initialize model, loss function, and optimizer
    model = InceptionModel(*DATA_DIM, NUM_OUTPUTS).cuda()

    # load the model weights from disk if it exists
    if LOAD_WEIGHT and os.path.exists('optimal_clip'):
        ckpt = torch.load('optimal_clip')
        model.load_state_dict(ckpt['state_dict'])
        print("Loaded Weights!")

    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(input[0].data.cpu().numpy())

    model.end_pool.register_forward_hook(hook_feature)    

    data_cpu, label = next(loader)

    if args.dx is not None:
        while label[0][int(args.dx)] != 1:
            data_cpu, label, non_imaging = next(loader)


    # convert data to cuda because model is cuda
    data, label = data_cpu.cuda(), label.type(torch.LongTensor).cuda()

    # eval mode changes behavior of dropout and batch norm for validation
    
    model.eval()
    probs = model(data)

    # get class predictions
    label = torch.argmax(label, dim = 1)
    preds = torch.argmax(model.softmax(probs), dim = 1)
    print(preds.item(), label.item())

    h_x = model.softmax(probs).data.squeeze()
    probs, idx = h_x.sort(0, True)

    bs, nc, h, w, d = features_blobs[0].shape
    conv_features = features_blobs[0][0]

    # compute smaller mask
    '''
    smaller_mat = skimage.transform.resize(data_cpu[0], (4, 8, 8), anti_aliasing = True, preserve_range = True, mode = 'edge')
    smaller_mat = smaller_mat - np.min(smaller_mat)
    smaller_mat = smaller_mat / np.max(smaller_mat)
    smaller_mat = np.uint8(255 * smaller_mat)
    small_mask = smaller_mat < 150
    '''

    # display downsampled brain mri
    '''
    fig, axes = plt.subplots(1, 4)
    for i in range(len(smaller_mat)):
        axes[i].imshow(smaller_mat[i])
    plt.pause(0.0001)
    '''

    # copy conv features for different tests
    '''
    conv_features_orig = conv_features
    conv_features_clip = copy.deepcopy(conv_features)
    '''

    # clip heatmap manually
    '''
    for i in range(len(conv_features_clip)):
        conv_features_clip[i][small_mask] = 0
    '''

    # resize conv features for cam
    '''
    conv_features_orig = conv_features_orig.reshape((nc, h*w*d))
    conv_features_clip = conv_features_clip.reshape((nc, h*w*d))
    '''
    conv_features = conv_features.reshape((nc, h*w*d))
    linear_features = np.squeeze(model.fc.weight.cpu().data.numpy())
   
    # get cam for clipped and unclipped
    '''
    orig = get_cam(conv_features_orig, linear_features, idx, (h,w,d), (128, 128, 64))
    clipped = get_cam(conv_features_clip, linear_features, idx, (h,w,d), (128, 128, 64))
    '''

    clipped = get_cam(conv_features, linear_features, idx, (h,w,d), (128, 128, 64))
    in_mat = skimage.transform.resize(data_cpu[0], (128, 128, 64), anti_aliasing = True, preserve_range = True, mode = 'edge')

    # Clip heatmap perfectly
    '''
    large_mask = in_mat == 0
    clipped[large_mask] = 0
    '''
    
    # show slice from a single POV
    '''
    fig, axes = plt.subplots(1, 3)
    for i in range(clipped.shape[2]):
        axes[0].cla()
        axes[1].cla()
        axes[2].cla()

        axes[0].imshow(in_mat[:,:, i])

        axes[1].imshow(in_mat[:, :, i])
        axes[1].imshow(orig[:,:, i], cmap='jet', alpha = 0.5)

        axes[2].imshow(in_mat[:, :, i])
        axes[2].imshow(clipped[:, :, i], cmap='jet', alpha=0.5)

        plt.pause(0.0001)
    plt.show()          
    '''

    if not os.path.exists('cam'):
        os.makedirs('cam')

    names = ["axial", "coronal", "sagittal"]
    paths = []
    fig, axes = plt.subplots(1, len(in_mat.shape))
    for i in range(in_mat.shape[1]):

        for j in range(len(in_mat.shape)):
            axes[j].cla()

            axes[j].axis('off')
            axes[j].tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')


        subscans = [
                    in_mat[min(i, in_mat.shape[0]) - 1, :, :],
                    in_mat[:, min(i, in_mat.shape[1]) - 1, :],
                    in_mat[:, :, min(i, in_mat.shape[2] - 1)]
                    ]

        clip_scans = [
                    clipped[min(i, clipped.shape[0]) - 1, :, :],
                    clipped[:, min(i, clipped.shape[1]) - 1, :],
                    clipped[:, :, min(i, clipped.shape[2] - 1)]
                    ]

        for j in range(len(in_mat.shape)):            
            axes[j].set_title(names[j])
            axes[j].imshow(subscans[j])
            axes[j].imshow(clip_scans[j], cmap='jet', alpha=0.5)

        #plt.pause(0.0001)

        filename = os.path.join('cam', 'frame_%d.png' % i)
        plt.savefig(filename)
        paths.append(filename)

    height, width, depth = cv2.imread(paths[0]).shape
    video = cv2.VideoWriter(os.path.join('cam', 'class activation map.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    for file in paths:
        video.write(cv2.imread(file))

    video.release()

    #plt.show()          


if __name__ == '__main__':
    main()

