import torch
import os
import numpy as np
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self,root,mode,patch_size, num_bins, num_edges,patch_nearest):
        self.root = root
        print('Dataset for', mode)
        self.patch_root = self.root + 'rotated_patches_{}_nearest_{}/'.format(patch_size,patch_nearest)
        self.pair_root = self.root + 'hist_pairwise_bins_{}_edges_{}/'.format(num_bins, num_edges)
        self.vertical_root = self.root + 'rotated_vertical_patches/'
        self.mode = mode
        self.size = 0
        self.patch_data_path = self.patch_root + self.mode
        self.vertical_data_path = self.vertical_root + self.mode
        self.pair_data_path = self.pair_root + self.mode
        self.neuron_list = os.listdir(self.patch_data_path)
        self.name_list = []
        for i in range(len(self.neuron_list)):
            sub_root = self.patch_data_path +'/'+ self.neuron_list[i]+'/'
            for j in range(len(os.listdir(sub_root))):
                self.name_list.append(os.listdir(sub_root)[j])
                self.size += 1

    def __getitem__(self, idx):
        label = int(self.name_list[idx].split('_')[-1][:-4])
        patch_npy_path = self.patch_data_path +'/'+ 'neuron_'+str(label) +'/'+self.name_list[idx]
        vertical_npy_path = self.vertical_data_path +'/'+ 'neuron_'+str(label) +'/'+self.name_list[idx]
        pair_npy_path = self.pair_data_path +'/'+ 'neuron_'+str(label) +'/'+self.name_list[idx]
        vertical_patch_data = torch.from_numpy((np.load(vertical_npy_path)).astype(np.float32))
        patch_data = torch.from_numpy((np.load(patch_npy_path)).astype(np.float32))
        pair_data = torch.from_numpy((np.load(pair_npy_path)).astype(np.float32))
        return patch_data, vertical_patch_data, pair_data, label

    def __len__(self):
        return self.size