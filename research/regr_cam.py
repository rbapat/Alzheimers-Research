# /home/rohan_bapat/Alzheimers-Research/research/cam/class activation map.avi

from torch.utils.data import DataLoader
import cv2
import torch.optim as optim
import multiprocessing
import numpy as np
import argparse
import torch.nn as nn
import torch
import shutil
import skimage
import matplotlib.pyplot as plt
import os

from research.datasets.scored_dataset import DataParser
from research.util.Grapher import TrainGrapher
from research.models.densenet import DenseNet

DATA_DIM = (128, 128, 128)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", help="which class (0 = CN, 1 = AD)")
    args = parser.parse_args()

    dataset = DataParser(DATA_DIM)
    num_outputs = dataset.get_num_outputs()

    loader = DataLoader(dataset.get_subset(1), batch_size = 1, shuffle = True)
        
    model = DenseNet(DATA_DIM, num_outputs, [6, 12, 24, 16], growth_rate = 12, theta = 1.0, drop_rate = 0.0).cuda()

    with torch.no_grad():
        ckpt = torch.load('optimal.t7')
        for name, param in ckpt['state_dict'].items():
            if name not in model.state_dict():
                print(name, "not it")
                continue

            if model.state_dict()[name].shape != param.shape:
                print("Failed shape", name)
                continue

            model.state_dict()[name].copy_(param)
            model.state_dict()[name].requires_grad = False

            #print("Loaded", name)


        print("Pretrained Weights Loaded!")

    criterion = nn.MSELoss()

    features = []
    gradients = []

    def save_conv(module, input, output):
        features.append(output.data.cpu().squeeze().numpy())

    def save_grad(module, grad_input, grad_output):
        gradients.append(grad_input[0].data.cpu().squeeze().numpy())

    model.model[0].layers[-1].register_backward_hook(save_grad)
    model.model[0].layers[-1].register_forward_hook(save_conv)

    loader = iter(loader)
    data, label = next(loader)
    data, label = data.cuda(), label.float().cuda()
    
    model.train(False)
    preds = model(data)

    model.zero_grad()
    preds.backward()

    conv_features = features[0]
    grads = gradients[0]

    weights = np.mean(grads, axis = (1,2,3))

    cam = np.zeros(conv_features.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_features[i, :, :, :]
    
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cam = skimage.transform.resize(cam, (128, 128, 128), anti_aliasing = True, preserve_range = True, mode = 'edge')

    in_mat = skimage.transform.resize(data[0].cpu(), (128, 128, 128), anti_aliasing = True, preserve_range = True, mode = 'edge')

    if not os.path.exists('cam'):
        os.makedirs('cam')

    names = ["sagittal", "coronal", "axial"]
    paths = []
    fig, axes = plt.subplots(1, len(in_mat.shape))
    for i in range(in_mat.shape[1]):

        for j in range(len(in_mat.shape)):
            axes[j].cla()

            axes[j].axis('off')
            axes[j].tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')


        subscans = [
                    np.rot90(in_mat[min(i, in_mat.shape[0]) - 1, :, :]),
                    np.rot90(in_mat[:, min(i, in_mat.shape[1]) - 1, :]),
                    np.rot90(in_mat[:, :, min(i, in_mat.shape[2] - 1)])
                    ]

        clip_scans = [
                    np.rot90(cam[min(i, cam.shape[0]) - 1, :, :]),
                    np.rot90(cam[:, min(i, cam.shape[1]) - 1, :]),
                    np.rot90(cam[:, :, min(i, cam.shape[2] - 1)])
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

    print("Predicted Score:", round(preds[0][0].item(), 4))
    print("Actual Score:", round(label[0][0].item(), 4))

if __name__ == '__main__':
    main()
