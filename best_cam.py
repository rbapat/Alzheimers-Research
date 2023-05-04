from scipy.ndimage import gaussian_filter
import xml.etree.ElementTree as ET
import torch.nn.functional as F
import torchvision.transforms.functional as T
import random
import re
from datetime import datetime
from models import *
import matplotlib as mpl
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
from collections import defaultdict

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

    DATASET_PATH = '/media/rohan/ThirdHardDrive/Research/Combined_FSL/scans'
    # DATASET_PATH = '/home/jupyter/Combined_FSL'

    EMBEDDING_PATH = '/media/rohan/ThirdHardDrive/Research/Combined_FSL/old_embeddings_288'
    # EMBEDDING_PATH = '/home/jupyter/Embedding'
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
            positive_grad_out = torch.clamp(grad_in[0], min=0.0)
            new_grad_in = positive_grad_out * grad

            return (new_grad_in,)

        for module in get_children(self):
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_full_backward_hook(backward_hook_fn)

    def load_weights(self):
        self.embedding_net.load_weights('ckpt.t7')
        self.predictor_net.load_state_dict(torch.load('predictor.t7'))#['state_dict'])

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

def save_heatmap(cam, guided_grads, orig_vols, ptidx, conv_view, label):    
    cam = skimage.transform.resize(cam.numpy(), (3, 182, 218, 182), anti_aliasing = True, preserve_range = True, mode = 'edge')
    cam = torch.from_numpy(cam)
    #cam = (cam - cam.mean())/cam.std()
    cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam))
        
    guided_grads = (guided_grads - guided_grads.mean())/guided_grads.std()

    guided_cam = cam * guided_grads
    guided_cam = torch.from_numpy(gaussian_filter(guided_cam, 1))

    mean_vol = torch.mean(guided_cam, axis = 0)
    mean_orig = torch.mean(orig_vols, axis = 0)

    path = os.path.join('heatmaps', 'volumes', str(conv_view), f'pt_{ptidx}_{label}.npy')
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok = True)

    with open(path, 'wb') as f:
        np.save(f, mean_vol)
        np.save(f, mean_orig)

    print(f"saved {path}")
        
def write_cam_volumes(c):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    parser = dataset.DataParser(c) 

    model = CAMNet(c).cuda()
    model.load_weights()
    model.embedding_net.eval()
    model.predictor_net.eval()
    criterion = nn.CrossEntropyLoss()


    #features = torch.zeros(3, 24, 23, 27, 23)
    #gradients = torch.zeros(3, 24, 23, 27, 23)

    features, gradients = [], []
    def save_conv(module, input, output):
        features.append(output.detach().cpu())
        #features.copy_(output.detach())

    def save_grad(module, grad_input, grad_output):
        #gradients.copy_(F.relu(grad_output[0].detach()))
        gradients.insert(0, F.relu(grad_output[0].detach()).cpu())

    for i in range(4):
        model.embedding_net.model[2*i].layers[0].register_full_backward_hook(save_grad)
        model.embedding_net.model[2*i].layers[0].register_forward_hook(save_conv)

    for idx, (volume_paths, clin_vars, dx) in enumerate(parser.full_testing_set()):
        # transform volume path into model input
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

        # set up model and pass volume in
        model.zero_grad()
        batch_volumes.requires_grad_()
        clin_vars.requires_grad_()
        raw_output = model(batch_volumes, clin_vars)

        # get loss and backprop gradient
        loss = criterion(raw_output, ground_truth)
        preds = torch.argmax(F.softmax(raw_output, dim = 1), dim = 1)
        is_correct = (preds == ground_truth).sum().item()
        
        one_hot_output = torch.FloatTensor(1, 2).zero_().cuda()
        one_hot_output[0][ground_truth[0].item()] = 1
        
        raw_output.backward(gradient = one_hot_output)
        #features, gradients = features[-1], gradients[-1]
        guided_grads = batch_volumes.grad[0, :, :, :, :].cpu()

        views = list(range(len(features)))
        for conv_view in views:
            f, g = features[conv_view], gradients[conv_view]
            weights = torch.mean(g, axis = (2, 3, 4))        
            cam = torch.zeros((3, *f.shape[2:]), dtype = torch.float32)
            for j in range(3):
                for i, w in enumerate(weights[j]):
                    cam[j] += F.relu(w * f[j, i, :, :, :])

            save_heatmap(cam, guided_grads, orig_volumes[0], idx, conv_view, f"{is_correct} | {idx}")

        features, gradients = [], []
        del raw_output

def get_scan_paths():
    base_path = os.path.join('heatmaps', 'volumes')
    conv_views = [path for path in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, path))]

    paths = {view: {} for view in conv_views}
    
    for view in conv_views:
        for (root, dirs, files) in os.walk(os.path.join('heatmaps', 'volumes', view)):
            for file in files:
                if file.endswith('.npy'):
                    paths[view][file] = os.path.join(root, file)

    return {'0': paths['0']}


def read_heatmap(path):
    with open(path, 'rb') as f:
        cam = np.load(f)
        orig = np.load(f)

    cam[cam < 0.3] = 0
    cam[cam > 1.0] = 1
    return torch.from_numpy(orig), torch.from_numpy(cam)

def render_scans(scan_list, view_name, view_num, thresh = 0.6):
    slices = [i / 10 for i in range(1, 10)]

    '''
    fig, axes = plt.subplots(len(scan_list), len(slices), constrained_layout = True)
    fig.set_figwidth(15)
    fig.set_figheight(7)
    fig.suptitle(view_name)
    '''

    fig = plt.figure(figsize=(10, 10), constrained_layout = True)
    gs = fig.add_gridspec(len(scan_list), len(slices), hspace=0, wspace=0)
    axes = gs.subplots(sharex='col', sharey='row')
    fig.suptitle(view_name)

    colormap = plt.cm.get_cmap("YlOrRd").copy()
    colormap.set_under('k', alpha = 0)

    #for idx, slice_num in enumerate(slices):
    #    axes[0][idx].set_title(str(slice_num))

    for pt_idx in range(len(scan_list)):
        orig, cam = scan_list[pt_idx][0], scan_list[pt_idx][1]

        for slice_idx, pct in enumerate(slices):
            slice_num = int(pct * cam.shape[view_num])
            ax = axes[pt_idx][slice_idx]

            im = np.rot90(cam.select(view_num, slice_num), 3)
            orig_im = np.rot90(orig.select(view_num, slice_num), 3)

            im = gaussian_filter(im, 2)

            ax.axis('off')
            ax.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

            ax.imshow(orig_im, cmap='gray', origin='lower')
            ax.imshow(im, cmap=colormap, origin='lower', clim=[0.3,1], alpha=0.7, interpolation='none')

def plot_scan(mean_vol, mean_orig, thresh = 0.6):
    slices = [i / 10 for i in range(1, 10)]

    z = torch.zeros_like(mean_vol)
    mean_vol = torch.where(mean_vol > thresh, mean_vol, z)

    colormap = plt.cm.get_cmap("jet").copy()
    colormap.set_under('k', alpha = 0)

    fig, axes = plt.subplots(3, len(slices))

    fig.set_figwidth(15)
    fig.set_figheight(7)

    for i, slice_num in enumerate(slices):
        axes[0][i].set_title(str(slice_num))

        for j in range(3):
            axes[j][i].axis('off')
            axes[j][i].tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

    for view in range(3):
        for i, slice_pct in enumerate(slices):
            slice_num = int(slice_pct * mean_vol.shape[view])
            ax = axes[view][i]

            im = np.rot90(mean_vol.select(view, slice_num), 3)
            orig_im = np.rot90(mean_orig.select(view, slice_num), 3)

            ax.axis('off')
            ax.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

            ax.imshow(orig_im, cmap='gray', origin='lower')
            ax.imshow(im, cmap=colormap, origin='lower', clim=[0.5,1.5], alpha=0.7, interpolation='hamming')

def plot_scan_new(mean_vol, mean_orig):
    slices = [i / 10 for i in range(2, 9)]

    colormap = plt.cm.get_cmap("YlOrRd").copy()
    colormap.set_under('k', alpha = 0)

    fig, axes = plt.subplots(3, len(slices))

    fig.set_figwidth(15)
    fig.set_figheight(7)

    for i, slice_num in enumerate(slices):
        axes[0][i].set_title(str(slice_num))

        for j in range(3):
            axes[j][i].axis('off')
            axes[j][i].tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

    for view in range(3):
        for i, slice_pct in enumerate(slices):
            slice_num = int(slice_pct * mean_vol.shape[view])
            ax = axes[view][i]

            im = np.rot90(mean_vol.select(view, slice_num), 3)
            orig_im = np.rot90(mean_orig.select(view, slice_num), 3)

            im = gaussian_filter(im, 2)

            ax.axis('off')
            ax.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

            ax.imshow(orig_im, cmap='gray', origin='lower')
            ax.imshow(im, cmap=colormap, origin='lower', clim = [0.3, 1], alpha=0.7, interpolation='none')

def create_atlas_cmap(atlas):
    color_list = []
    cmap_lut = {}
    
    unique_colors = torch.unique(atlas)
    jet_cmap = plt.cm.get_cmap("jet").copy()
    upper = torch.max(unique_colors)
    for color_id in unique_colors:
        col = jet_cmap(int((color_id / upper) * (jet_cmap.N-1)))
        # col = list(np.random.randn(3))
        # col.append(1)
        color_list.append(col)

        col = tuple([round(i, 4) for i in col])
        cmap_lut[col] = int(color_id.item())
    
    color_list[0] = (0, 0, 0, 1.0)
    cmap_lut[(0, 0, 0, 1.0)] = 0

    return cmap_lut, mpl.colors.LinearSegmentedColormap.from_list('Atlas cmap', color_list, len(color_list))

def explain_cam(orig, cam, atlas_path, atlas_xml):
    labels = {}
    atlas = torch.from_numpy(nib.load(atlas_path).get_fdata()).to(torch.int)

    total_counts = torch.bincount(atlas.flatten())

    atlas[cam < 0.3] = 0

    xml_labels = ET.parse(atlas_xml).getroot().findall('./data/label')
    atlas_labels = {int(el.attrib['index'])+1: el.text for el in xml_labels}
    atlas_labels[0] = 'nothing'
    
    # whitelist = [   
    #     'Temporal Pole', 
    #     'Middle Temporal Gyrus, posterior division', 
    #     'Parahippocampal Gyrus, anterior division', 
    #     'Inferior Temporal Gyrus, posterior division', 
    #     'Temporal Fusiform Cortex, posterior division', 
    #     'Middle Temporal Gyrus, anterior division',
    #     'Insular Cortex',
    #     'Planum Polare'
    # ]

    # for lab_idx in atlas_labels:
    #     if atlas_labels[lab_idx] not in whitelist and atlas_xml != "/home/rohan/fsl/data/atlases/HarvardOxford-Subcortical.xml":
    #         atlas[atlas == lab_idx] = 0

    counts = torch.bincount(atlas.flatten())
    if len(total_counts) > len(counts):
        new_counts = torch.zeros_like(total_counts)
        new_counts[:len(counts)] = counts
        counts = new_counts

    if len(total_counts) != len(counts):
        print(total_counts.unique())
        print(counts.unique())
        raise RuntimeError(f'len(total_counts) != len(counts) => {len(total_counts)} != {len(counts)}')

    # if len(counts) != len(atlas_labels):
    #     raise RuntimeError(f'len(counts) != len(atlas_labels) => {len(counts)} != {len(atlas_labels)}')

    # for idx, (subcnt, totcnt) in enumerate(zip(counts, total_counts)):
    #     print(f'{atlas_labels[idx]}: {subcnt} / {totcnt} = {subcnt / totcnt}')
    # input()


    hist = [(atlas_labels[idx], cnt) for idx, cnt in enumerate(counts)]
    hist.sort(key = lambda x: x[1], reverse = True)

    # cmap_lut, cmap = create_atlas_cmap(atlas)

    # fig = plot_atlas(orig, atlas, cmap)

    # ax = plt.subplot(1, 11, 11)
    # cb = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap),cax=ax, orientation='vertical') 
    
    # color_ids = torch.unique(atlas)
    # ticks = np.linspace(0, 1, len(color_ids)+1)+(1/len(color_ids)/2)
    # ticks = ticks[:-1]

    # labels = []
    # for i in ticks:
    #     col = tuple([round(i, 4) for i in cmap(i)])
    #     labels.append(atlas_labels[cmap_lut[col]])

    # cb.set_ticks(ticks)
    # cb.set_ticklabels(labels) 

    return hist, counts / total_counts


def plot_atlas(orig, atlas, cmap): # sub_atlas.nii.gz
    slices = [i / 10 for i in range(1, 10)]
    plt.figure(figsize=(16, 8)) 

    index = 1
    for view in range(3):
        for i, slice_pct in enumerate(slices):
            ax = plt.subplot(3, len(slices), index)
            ax.axis('off')
            ax.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

            slice_num = int(slice_pct * atlas.shape[view])

            im = np.rot90(atlas.select(view, slice_num), 3)
            orig_im = np.rot90(orig.select(view, slice_num), 3)

            ax.axis('off')
            ax.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

            ax.imshow(orig_im, cmap='gray', origin='lower')
            ax.imshow(im, cmap=cmap, origin='lower', alpha=0.7, interpolation='none')

            index += 1    

def plot_heatmaps(path_mapping, num_to_render = 5):
    regex_str = r"pt_([-+]?[0-9]+)_([-+]?[0-9]+) | ([-+]?[0-9]+).npy" 
    
    atlases = [
        ('/home/rohan/fsl/data/atlases/MNI/MNI-maxprob-thr25-1mm.nii.gz', '/home/rohan/fsl/data/atlases/MNI.xml'),
        ('/home/rohan/fsl/data/atlases/Cerebellum/Cerebellum-MNIflirt-maxprob-thr25-1mm.nii.gz', '/home/rohan/fsl/data/atlases/Cerebellum_MNIflirt.xml'),
        ('/home/rohan/fsl/data/atlases/Talairach/Talairach-labels-1mm.nii.gz', '/home/rohan/fsl/data/atlases/Talairach.xml'),
        ('/home/rohan/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz', '/home/rohan/fsl/data/atlases/HarvardOxford-Subcortical.xml'),
        ('/home/rohan/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz', '/home/rohan/fsl/data/atlases/HarvardOxford-Cortical.xml'),
        
        
    ]

    

    for view in path_mapping:
        # correct_paths = list(filter(lambda p: re.search(regex_str, p).groups()[1] == '1', list(path_mapping[view].keys())))
        # filenames = random.choices(correct_paths, k = num_to_render)

        # scans = [read_heatmap(path_mapping[view][fname]) for fname in filenames]
        # views = ['Sagittal', 'Coronal', 'Axial']
        
        # for view_idx, view_name in enumerate(views):
        #     render_scans(scans, view_name, view_idx)

        #     path = os.path.join('heatmaps', 'images', view, f"{view_name}.png")
        #     dirname = os.path.dirname(path)

        #     if not os.path.exists(dirname):
        #         os.makedirs(dirname, exist_ok = True)

        #     plt.savefig(path)
        #     print(f'saved {path}')
        
        avg_cam = torch.zeros(182, 218, 182)
        orig_cam = torch.zeros(182, 218, 182)
        pt_freqs = {os.path.basename(path[1]): defaultdict(int) for path in atlases}
        pt_pcts = {os.path.basename(path[1]): defaultdict(int) for path in atlases}

        for filename in path_mapping[view]:
            orig, cam = read_heatmap(path_mapping[view][filename])

            # avg_cam += torch.where(cam > 0, cam, 0) / len(path_mapping[view]) # real thresholding
            avg_cam += torch.where(cam > 0, 1, 0) / len(path_mapping[view]) # binarized version

            orig_cam[:] = orig

            # plot_scan_new(cam, orig)

            # path = os.path.join('heatmaps', 'images', view, filename.replace('npy', 'png'))
            # dirname = os.path.dirname(path)
            # if not os.path.exists(dirname):
            #     os.makedirs(dirname, exist_ok = True)

            # plt.savefig(path)
            # print(f'saved {path}')

            for atlas_path, atlas_xml in atlases:
                name = os.path.basename(atlas_path).replace('.nii.gz', '')
                path = os.path.join('heatmaps', 'images', view, name, filename.replace('npy', 'png'))

                dirname = os.path.dirname(path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname, exist_ok = True)

                hist, pct = explain_cam(orig, cam, atlas_path, atlas_xml)

                dct = pt_freqs[os.path.basename(atlas_xml)]
                for (region, count), _pct in zip(hist, pct):
                    if region != 'nothing' and count > 0 and _pct > 0.2:
                        dct[region] += 1
                

                plt.tight_layout()
                plt.savefig(path , bbox_inches='tight')
                # print(f'saved {path}')
            plt.figure()       
        
        # BAD CODE, BIG REFACTOR NEED!
        plot_scan_new(avg_cam, orig_cam)

        path = os.path.join('heatmaps', 'images', view, 'avg.png')
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok = True)

        plt.tight_layout()
        plt.savefig(path)
        print(f'saved {path}')

        for atlas_path, atlas_xml in atlases:
            name = os.path.basename(atlas_path).replace('.nii.gz', '')
            path = os.path.join('heatmaps', 'images', view, name, 'avg.png')

            dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok = True)

            hist = explain_cam(orig_cam, avg_cam, atlas_path, atlas_xml)
            
            plt.tight_layout()
            plt.savefig(path , bbox_inches='tight')
            print(f'saved {path}')

        txt_path = os.path.join('heatmaps', 'images', 'regions.txt')
        with open(txt_path, 'w') as f:
            f.write(f'View {view}\n')
            for atlas in pt_freqs:
                dct_print = {k: v for k, v in sorted(pt_freqs[atlas].items(), reverse=True, key=lambda item: item[1])}
                f.write(f'Atlas: {atlas}\n')
                for region_name in dct_print:
                    f.write(f'{region_name}: {dct_print[region_name]}\n')
                f.write('\n\n')
            f.write('\n\n')
        
        

def write_cam_images(c):
    random.seed(0)
    paths = get_scan_paths()
    plot_heatmaps(paths)

def main(c):
    # write_cam_volumes(c)
    write_cam_images(c)

if __name__ == '__main__':
    main(Constants())
