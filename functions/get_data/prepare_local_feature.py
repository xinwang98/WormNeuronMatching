import numpy as np
import os
from functions.get_data.get_patch.get_patch import get_patch

def prepare_local_feature(patch_size, num_nearest_neurons, head_neuron=77, tail_neuron=1, train_val_border=40,
                          data_root='./data/npy_files/', save_root='./data/ordered_dataset/'):

    patch_root = save_root + 'rotated_patches_{}_nearest_{}/'.format(patch_size,num_nearest_neurons)
    vertical_patch_root = save_root + 'rotated_vertical_patches/'
    flag = True
    if not os.path.exists(patch_root):
        os.makedirs(patch_root)
        flag = False
    if not os.path.exists(vertical_patch_root):
        os.makedirs(vertical_patch_root)
        flag = False
    if flag:
        return
    positions = np.load(os.path.join(data_root, 'ordered_neuron_position.npy'))
    images = np.load(os.path.join(data_root, 'img_stack.npy'))
    tail_pos = positions[:, tail_neuron, :]
    head_pos = positions[:, head_neuron, :]
    patches, vertical_patch  = get_patch(frames=images, neuron_position=positions, head_pos=head_pos,
                                         tail_pos=tail_pos, patch_size=patch_size, num_nearest_neurons=num_nearest_neurons)

    vertical_patch = vertical_patch[:, :, :, :, np.newaxis]    # add the channel dim
    patch_mean, vertical_mean = np.mean(patches, axis=(0, 1)), np.mean(vertical_patch, axis=(0, 1))
    patches, vertical_patch = (patches - patch_mean), (vertical_patch - vertical_mean)
    patches, vertical_patch = np.transpose(patches, (0, 1, 4, 2, 3)), np.transpose(vertical_patch, (0, 1, 4, 2, 3))
    patches, vertical_patch = patches.astype(np.float32), vertical_patch.astype(np.float32)

    num_frame, num_neuron = patches.shape[0:2]
    for frame_idx in range(num_frame):
        for neuron_idx in range(num_neuron):
            patch_npy = patches[frame_idx,neuron_idx]
            vertical_patch_npy = vertical_patch[frame_idx,neuron_idx]
            if frame_idx < train_val_border:
                patch_path = patch_root + 'train/neuron_'+ str(neuron_idx) + '/'
                vertical_patch_path = vertical_patch_root + 'train/neuron_'+ str(neuron_idx) + '/'
                if not os.path.exists(patch_path):
                    os.makedirs(patch_path)
                if not os.path.exists(vertical_patch_path):
                    os.makedirs(vertical_patch_path)
                patch_npy_name = patch_path + 'frame_{}_neuron_{}'.format(frame_idx,neuron_idx)
                vertical_patch_npy_name = vertical_patch_path + 'frame_{}_neuron_{}'.format(frame_idx, neuron_idx)
            else :
                patch_path = patch_root + 'val/neuron_' + str(neuron_idx) + '/'
                vertical_patch_path = vertical_patch_root + 'val/neuron_' + str(neuron_idx) + '/'
                if not os.path.exists(patch_path):
                    os.makedirs(patch_path)
                if not os.path.exists(vertical_patch_path):
                    os.makedirs(vertical_patch_path)
                patch_npy_name = patch_path + 'frame_{}_neuron_{}'.format(frame_idx, neuron_idx)
                vertical_patch_npy_name = vertical_patch_path + 'frame_{}_neuron_{}'.format(frame_idx, neuron_idx)
            np.save(patch_npy_name, patch_npy)
            np.save(vertical_patch_npy_name, vertical_patch_npy)

    print('Local data is ready!')
    print('---------------------')