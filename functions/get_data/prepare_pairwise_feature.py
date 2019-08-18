from sklearn.decomposition import PCA
import numpy as np
import os
from functions.get_data.get_pairwise.get_pairwise_feature import get_pairwise_feature



def prepare_hist_pairwise_feature(num_bins, num_edge, head_neuron, tail_neuron, train_val_border=40,
                                  file_root='./data/npy_files/', save_root='./data/ordered_dataset/'):
    neuron_position = np.load(file_root + 'ordered_neuron_position.npy')

    frame_num = neuron_position.shape[0]
    pairwise_dir = save_root + 'hist_pairwise_bins_{}_edges_{}'.format(num_bins, num_edge)
    if os.path.exists(pairwise_dir):
        return
    for frame_idx in range(frame_num):
        print('Frame {}'.format(frame_idx))
        tail2head = neuron_position[frame_idx, tail_neuron, 0:2] - neuron_position[frame_idx, head_neuron, 0:2]
        pairwise_hist_features, sphere_feature = get_pairwise_feature(
                                                    neuron_position[frame_idx, :, :].copy(), num_edge,
                                                    num_bins, tail2head)

        pairwise_features = np.concatenate((pairwise_hist_features, sphere_feature), axis=1)

        for neuron_idx in range(neuron_position.shape[1]):
            neuron_npy = pairwise_features[neuron_idx]
            if frame_idx < train_val_border:
                pairwise_path = save_root + 'hist_pairwise_bins_{}_edges_{}/train/neuron_{}/'.format(num_bins, num_edge,
                                                                                                neuron_idx)
                if not os.path.exists(pairwise_path):
                    os.makedirs(pairwise_path)
                pair_npy_name = pairwise_path + 'frame_{}_neuron_{}.npy'.format(frame_idx, neuron_idx)
            else:
                pairwise_path = save_root + 'hist_pairwise_bins_{}_edges_{}/val/neuron_{}/'.format(num_bins, num_edge,
                                                                                              neuron_idx)
                if not os.path.exists(pairwise_path):
                    os.makedirs(pairwise_path)
                pair_npy_name = pairwise_path + 'frame_{}_neuron_{}.npy'.format(frame_idx, neuron_idx)
            np.save(pair_npy_name, neuron_npy)

    print('pairwise data is ready!')
    print('---------------------')