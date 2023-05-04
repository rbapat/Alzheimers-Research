import torch.nn.functional as F
from datetime import datetime
import nibabel as nib
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import dataset
import random
import skimage
import torch
import os
import gc

from models import DenseNet, LSTMNet

BATCH_SIZE = 1
NUM_EPOCHS = 2000
IN_DIMS = (182, 218, 182)
OUTPUT_DIM = 2

class CAMNet(nn.Module):
    def __init__(self):
        super(CAMNet, self).__init__()

        self.embedding_net = DenseNet(IN_DIMS, OUTPUT_DIM, [6, 12, 32, 24], growth_rate = 24, theta = 0.5, drop_rate = 0.0)

        m3 = lambda lstm_hidden, out_dims: nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden // 2), nn.ReLU(), nn.Linear(lstm_hidden // 2, lstm_hidden // 4), nn.ReLU(), nn.Linear(lstm_hidden // 4, lstm_hidden // 8), nn.ReLU(), nn.Linear(lstm_hidden // 8, out_dims))
        m2 = lambda lstm_hidden, out_dims: nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden // 2), nn.ReLU(), nn.Linear(lstm_hidden // 2, lstm_hidden // 4), nn.ReLU(), nn.Linear(lstm_hidden // 4, out_dims))
        self.lstm = LSTMNet(1000, 1024, 2, m2, 0.1)

    def load_weights(self):
        self.embedding_net.load_weights('embedding.t7')
        self.lstm.load_state_dict(torch.load('lstm.t7')['state_dict'])

    def forward(self, x, clin_vars):
        bs, seq_len, d1, d2, d3 = x.shape

        x = x.view(bs * seq_len, d1, d2, d3)

        x = self.embedding_net.features(x)

        x = x.view(bs, seq_len, -1)

        x = self.lstm(x, clin_vars)
        return x

def transform_strings(volume_paths):
    # currently, volume_paths is (3, BATCH_SIZE), type = list(tuple)
    # I want it to be (BATCH_SIZE, 3)

    new_list = [[] for _ in range(BATCH_SIZE)]

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

    colormap = plt.cm.get_cmap("jet").copy()
    colormap.set_under('k', alpha = 0)

    fig, axes = plt.subplots(3, len(slices) + 1)
    fig.set_figwidth(15)
    fig.set_figheight(7)
    fig.suptitle(title)
    for idx, slice_num in enumerate(slices):
        axes[0][idx + 1].set_title(str(slice_num))

    for i, axis in enumerate(axes):
        axis[0].axis('off')
        axis[0].tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        axis[0].text(0.3, 0.5, f"{i * 6} Months")

        for idx, slice_num in enumerate(slices):
            axis[idx + 1].axis('off')
            axis[idx + 1].tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

            if view == 0:
                axis[idx + 1].imshow(np.rot90(orig_scans[i][slice_num, :, :]))
                axis[idx + 1].imshow(np.rot90(cam_scans[i][slice_num, :, :]), cmap='jet', alpha=0.4)    
            elif view == 1:
                axis[idx + 1].imshow(np.rot90(orig_scans[i][:, slice_num, :]))
                axis[idx + 1].imshow(np.rot90(cam_scans[i][:, slice_num, :]), cmap='jet', alpha=0.4)    
            elif view == 2:
                axis[idx + 1].imshow(np.rot90(orig_scans[i][:, :, slice_num]))
                axis[idx + 1].imshow(np.rot90(cam_scans[i][:, :, slice_num]), cmap='jet', alpha=0.4)    

    save_path = os.path.join('train_graphs', f'pt_{ptnum}', f'{view}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    plt.savefig(save_path)
    plt.clf()

def plot_heatmap(cam, orig_vols, label, thresh = 50):

    cam = cam - torch.min(cam)
    cam = 255 * (cam / torch.max(cam))
    cam = np.array(cam, dtype = np.uint8)
    cam = np.where(cam > thresh, cam, 0)
    cam = skimage.transform.resize(cam, (3, 182, 218, 182), anti_aliasing = True, preserve_range = True, mode = 'edge')

    for i in range(3):  
        render(orig_vols[:, :, :, :], cam, label, label.split(' ')[-1], i)

# plots class activation maps (heatmaps) for test set using pretrained models
def main():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    parser = dataset.DataParser(3, BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    model = CAMNet().cuda()
    model.load_weights()
    model.embedding_net.eval()
    model.lstm.train()

    features = torch.zeros(3, 24, 23, 27, 23)
    def save_conv(module, input, output):
        #features.append(output.data.cpu().squeeze().numpy())
        features.copy_(output.detach())

    gradients = torch.zeros(3, 24, 23, 27, 23)
    def save_grad(module, grad_input, grad_output):
        #gradients.append(torch.nn.functional.relu(grad_output[0].data.cpu()).squeeze().numpy())
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
    for idx, (volume_paths, clin_vars, dx) in enumerate(parser.loaders[1]):
        with torch.no_grad():
            volume_paths = transform_strings(volume_paths)
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
        raw_output = model(batch_volumes, clin_vars)

        loss = criterion(raw_output, ground_truth)
        preds = torch.argmax(F.softmax(raw_output, dim = 1), dim = 1)
        is_correct = (preds == ground_truth).sum().item()
        
        raw_output[0, preds[0].item()].backward()

        '''
        cam_scans = [draw_cam(feature, grad) for feature, grad in zip(features, gradients)]

        for i in range(3):
            render(orig_volumes[0, :, :, :, :], cam_scans, f"Predicted: {preds[0].item()}, Actual: {ground_truth[0].item()}", idx, i)
        '''

        weights = torch.mean(gradients, axis = (2, 3, 4))        
        cam = torch.zeros((3, *features.shape[2:]), dtype = torch.float32)
        for j in range(3):
            for i, w in enumerate(weights[j]):
                cam[j] += w * features[j, i, :, :, :]

        plot_heatmap(cam, orig_volumes[0], f"{is_correct} {idx}")


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

    plot_heatmap(corrects_map, avg_scan, "Mean Heatmap - Correct")
    plot_heatmap(incorrect_map, avg_scan, "Mean Heatmap - Incorrect")
    plot_heatmap(cvt_map, avg_scan, "Mean Heatmap - Converters")
    plot_heatmap(ncvt_map, avg_scan, "Mean Heatmap - Nonconverters")
    plot_heatmap(corrects_map, avg_scan, "Mean Heatmap - Combined")

    '''
    plot_heatmap(corrects_map, mean_orig_volumes[0], "Mean Heatmap - Correct")
    plot_heatmap(incorrect_map, mean_orig_volumes[0], "Mean Heatmap - Incorrect")
    plot_heatmap(cvt_map, mean_orig_volumes[0], "Mean Heatmap - Converters")
    plot_heatmap(ncvt_map, mean_orig_volumes[0], "Mean Heatmap - Nonconverters")
    plot_heatmap(corrects_map, mean_orig_volumes[0], "Mean Heatmap - Combined")
    '''
        
if __name__ == '__main__':
    main()




'''
# LSTM model that I'm using right now, relatively simple and can definitely be improved/tuned further
class LSTMNet(nn.Module):
    def __init__(self, conv_features = 1000, lstm_hidden = 512, lstm_layers = 2):
        super(LSTMNet, self).__init__()

        #conv_features += 4
        self.lstm = nn.LSTM(input_size = conv_features, hidden_size = lstm_hidden, num_layers = lstm_layers, bias = True, batch_first = True, dropout = 0.3)
        
        m0 = lambda lstm_hidden, out_dims: nn.Sequential(nn.Linear(lstm_hidden, out_dims))
        m1 = lambda lstm_hidden, out_dims: nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden // 2), nn.ReLU(), nn.Linear(lstm_hidden // 2, out_dims))
        m2 = lambda lstm_hidden, out_dims: nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden // 2), nn.ReLU(), nn.Linear(lstm_hidden // 2, lstm_hidden // 4), nn.ReLU(), nn.Linear(lstm_hidden // 4, out_dims))
        m3 = lambda lstm_hidden, out_dims: nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden // 2), nn.ReLU(), nn.Linear(lstm_hidden // 2, lstm_hidden // 4), nn.ReLU(), nn.Linear(lstm_hidden // 4, lstm_hidden // 8), nn.ReLU(), nn.Linear(lstm_hidden // 8, out_dims))

        self.predictor = m3(lstm_hidden, 2)

    def forward(self, x, clin_vars):
        #x = torch.cat([x, clin_vars], dim = 2)
        x, _ = self.lstm(x) # num_layers, batch_size, lstm_hidden
        x = self.predictor(x[:, -1, :]) 
        return x
'''