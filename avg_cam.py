import torch.nn.functional as F
from datetime import datetime
from models import *
import torch.nn as nn
import numpy as np
import argparse
import dataset
import logger
import random
import torch
import os
import nibabel as nib
import skimage.transform
import matplotlib.pyplot as plt

class Constants:
    CLASSIFICATION = (1 << 0)
    REGRESSION = (1 << 1)
    SINGLE_TIMEPOINT = (1 << 2)
    LONGITUDINAL = (1 << 3)

    # OPERATION = CLASSIFICATION | SINGLE_TIMEPOINT
    OPERATION = CLASSIFICATION | LONGITUDINAL
    DX_CAP = 2
    BATCH_SIZE = 1
    LEARNING_RATE = 0.0001      
    NUM_EPOCHS = 500
    IN_DIMS = (182, 218, 182)
    OUTPUT_DIM = 2
    CROSS_VAL = True
    INNER_SPLIT = 3
    OUTER_SPLIT = 5

    # DATASET_PATH = '/media/rohan/ThirdHardDrive/Combined_FSL'
    DATASET_PATH = '/home/jupyter/Combined_FSL'

    # EMBEDDING_PATH = '/home/rohan/Documents/Alzheimers/embeddings_288'
    EMBEDDING_PATH = '/home/jupyter/Embedding'
    SPLITS = [0.8, 0.2]
    LOAD_PATHS = True
    
    CLIN_VARS = ['MMSE', 'CDRSB', 'mPACCtrailsB', 'mPACCdigit', 'APOE4', 'ADAS11', 'ADAS13', 'ADASQ4', 'FAQ', 'RAVLT_forgetting', 'RAVLT_immediate', 'RAVLT_learning', 'TRABSCOR']
    VISIT_DELTA = 6
    NUM_VISITS = 3


def get_children(model):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children

class CAMNet(nn.Module):
    def __init__(self, c):
        super(CAMNet, self).__init__()

        self.embedding_net = DenseNet(c.IN_DIMS, c.OUTPUT_DIM, [6, 12, 32, 24], growth_rate = 24, theta = 0.5, drop_rate = 0.0).cuda()
        self.predictor_net = MultiModalNet(288, len(c.CLIN_VARS))
        self.activation_maps = []
        self.gradient = None


        def forward_hook_fn(module, input, output):
            self.activation_maps.append(output)

        def backward_hook_fn(module, grad_in, grad_out):
            grad = self.activation_maps.pop() 
            # for the forward pass, after the ReLU operation, 
            # if the output value is positive, we set the value to 1,
            # and if the output value is negative, we set it to 0.
            grad[grad > 0] = 1 
            
            # grad_out[0] stores the gradients for each feature map,
            # and we only retain the positive gradients
            positive_grad_out = torch.clamp(grad_out[0], min=0.0)
            new_grad_in = positive_grad_out * grad

            return (new_grad_in,)

        def hook_function(module, grad_in, grad_out):
            self.gradient =  grad_in[0]

        self.embedding_net.stem[0].model[0].register_full_backward_hook(hook_function)

        for module in get_children(self):
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_full_backward_hook(backward_hook_fn)

    def load_weights(self):
        self.embedding_net.load_weights('ckpt.t7')
        self.predictor_net.load_state_dict(torch.load('predictor.t7')['state_dict'])

    def forward(self, x, clin_vars):
        bs, seq_len, d1, d2, d3 = x.shape

        x = x.view(bs * seq_len, d1, d2, d3)

        x = self.embedding_net.features(x)

        x = x.view(bs, seq_len, -1)
        
        x = self.predictor_net(x, clin_vars)
        return x

def transform_strings(c, volume_paths):
    # currently, volume_paths is (3, BATCH_SIZE), type = list(tuple)
    # I want it to be (BATCH_SIZE, 3)

    new_list = [[] for _ in range(c.BATCH_SIZE)]

    for paths in volume_paths:
        for idx, path in enumerate(paths):
            new_list[idx].append(path)

    return new_list

def draw_cam(feature, grad):
    weights = torch.mean(grad, axis = (1, 2, 3))
    cam = torch.zeros(feature.shape[1:], dtype = torch.float32)

    for i, w in enumerate(weights):
        cam += w * feature[i, :, :, :]

    cam = cam.detach().cpu()
    cam = cam - torch.min(cam)
    cam = 255 * (cam / torch.max(cam))
    cam = np.array(cam, dtype = np.uint8)
    cam = np.where(cam > 50, cam, 0)
    cam = skimage.transform.resize(cam, (182, 218, 182), anti_aliasing = True, preserve_range = True, mode = 'edge')

    return cam

def render(orig_scans, cam_scans, title, ptnum, view):
    #slices = [30, 52, 74, 96, 128, 150, 172]
    slices = [50, 70, 90, 100, 120, 140, 150]

    mean_orig_scan = torch.mean(orig_scans, axis = 0)
    mean_cam_scans = np.mean(cam_scans, axis = 0)

    colormap = plt.cm.get_cmap("jet").copy()
    colormap.set_under('k', alpha = 0)

    fig, axes = plt.subplots(1, len(slices))
    fig.set_figwidth(15)
    fig.set_figheight(7)
    fig.suptitle(title)
    for i, slice_num in enumerate(slices):
        axes[i].set_title(str(slice_num))
        axes[i].axis('off')
        axes[i].tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

        if view == 0:
            axes[i].imshow(np.rot90(mean_orig_scan[slice_num, :, :]))
            axes[i].imshow(np.rot90(mean_cam_scans[slice_num, :, :]), cmap='jet', alpha=0.4)    
        elif view == 1:
            axes[i].imshow(np.rot90(mean_orig_scan[:, slice_num, :]))
            axes[i].imshow(np.rot90(mean_cam_scans[:, slice_num, :]), cmap='jet', alpha=0.4)    
        elif view == 2:
            axes[i].imshow(np.rot90(mean_orig_scan[:, :, slice_num]))
            axes[i].imshow(np.rot90(mean_cam_scans[:, :, slice_num]), cmap='jet', alpha=0.4)    

    save_path = os.path.join('train_graphs', f'pt_{ptnum}', f'{view}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    plt.savefig(save_path)
    plt.clf()

def plot_heatmap(cam, guided_grads, orig_vols, label, thresh = 50):

    cam = skimage.transform.resize(cam, (3, 182, 218, 182), anti_aliasing = True, preserve_range = False, mode = 'edge')

    if guided_grads is not None:
        guided_grads = (guided_grads - guided_grads.min()) / (guided_grads.max()-guided_grads.min())
        # cam *= guided_grads

    cam = cam - np.min(cam)
    cam = 255 * (cam / np.max(cam))
    cam = np.array(cam, dtype = np.uint8)
    cam = np.where(cam > thresh, cam, 0)

    for i in range(3):  
        render(orig_vols[:, :, :, :], cam, label, label.split(' ')[-1], i)
        
def main(c):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    parser = dataset.DataParser(c) 

    model = CAMNet(c).cuda()
    model.load_weights()
    model.embedding_net.eval()
    model.predictor_net.eval()
    criterion = nn.CrossEntropyLoss()


    features = torch.zeros(3, 24, 23, 27, 23)
    gradients = torch.zeros(3, 24, 23, 27, 23)

    def save_conv(module, input, output):
        features.copy_(output.detach())

    def save_grad(module, grad_input, grad_output):
        gradients.copy_(F.relu(grad_output[0].detach()))

    model.embedding_net.model[0].layers[0].register_full_backward_hook(save_grad)
    model.embedding_net.model[0].layers[0].register_forward_hook(save_conv)

    corrects = 0
    
    cor, incor, cvt, ncvt = 0, 0, 0, 0
    
    cvt_map = torch.zeros((3, *features.shape[2:]), dtype = torch.float32)
    ncvt_map = torch.zeros((3, *features.shape[2:]), dtype = torch.float32)
    corrects_map = torch.zeros((3, *features.shape[2:]), dtype = torch.float32)
    incorrect_map = torch.zeros((3, *features.shape[2:]), dtype = torch.float32)
    combined_map = torch.zeros((3, *features.shape[2:]), dtype = torch.float32)
    avg_scan = torch.zeros((3, 182, 218, 182), dtype = torch.float32)

    mean_orig_volumes = None
    for idx, (volume_paths, clin_vars, dx) in enumerate(parser.full_testing_set()):
        with torch.no_grad():
            volume_paths = transform_strings(c, volume_paths)
            clin_vars, ground_truth = clin_vars.cuda(), dx.view(dx.shape[0]).cuda()

            batch_volumes = []
            for path_batch in volume_paths:
                volumes = []
                for path in path_batch:
                    mat = nib.load(path).get_fdata()
                    mat = (mat - mat.min()) / (mat.max() - mat.min())        
                    mat = torch.Tensor(np.array(mat)).unsqueeze(0).cuda()

                    volumes.append(mat)

                volumes = torch.cat(volumes).unsqueeze(0)
                batch_volumes.append(volumes)

            batch_volumes = torch.cat(batch_volumes)
            orig_volumes = batch_volumes.detach().cpu().clone()
            if mean_orig_volumes is None:
                mean_orig_volumes = orig_volumes

        avg_scan += orig_volumes[0, :, :, :, :]
        model.zero_grad()

        batch_volumes.requires_grad_()
        clin_vars.requires_grad_()
        raw_output = model(batch_volumes, clin_vars)

        loss = criterion(raw_output, ground_truth)
        preds = torch.argmax(F.softmax(raw_output, dim = 1), dim = 1)
        is_correct = (preds == ground_truth).sum().item()
        
        one_hot_output = torch.FloatTensor(1, 2).zero_().cuda()
        one_hot_output[0][ground_truth[0].item()] = 1
        
        raw_output.backward(gradient = one_hot_output)

        weights = torch.mean(gradients, axis = (2, 3, 4))        
        cam = torch.zeros((3, *features.shape[2:]), dtype = torch.float32)
        for j in range(3):
            for i, w in enumerate(weights[j]):
                cam[j] += w * features[j, i, :, :, :]

        cam = F.relu(cam)
        guided_grads = model.gradient.data[:, 0, :, :, :].cpu().numpy()
        plot_heatmap(cam, guided_grads, orig_volumes[0], f"{is_correct} {idx}")


        combined_map += cam
        if is_correct:
            cor += 1
            corrects_map += cam
        else:
            incor += 1
            incorrect_map += cam

        if ground_truth[0].item():
            cvt += 1
            cvt_map += cam
        else:
            ncvt += 1
            ncvt_map += cam

        print(cor + incor)
        del raw_output


    corrects_map /= cor
    incorrect_map /= incor
    cvt_map /= cvt
    ncvt_map /= ncvt
    combined_map /= (cor + incor)
    avg_scan /= (cor + incor)

    plot_heatmap(corrects_map, None, avg_scan, "Mean Heatmap - Correct")
    plot_heatmap(incorrect_map,  None, avg_scan, "Mean Heatmap - Incorrect")
    plot_heatmap(cvt_map,  None, avg_scan, "Mean Heatmap - Converters")
    plot_heatmap(ncvt_map,  None, avg_scan, "Mean Heatmap - Nonconverters")
    plot_heatmap(corrects_map,  None, avg_scan, "Mean Heatmap - Combined")

if __name__ == '__main__':
    main(Constants())
