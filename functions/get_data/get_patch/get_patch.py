import numpy as np
from sklearn.decomposition import PCA
from .rotation_crop import rotation_crop
import os


def get_patch(frames, neuron_position, head_pos, tail_pos, patch_size=64, num_nearest_neurons=50,
              file_root='./data/patch_dataset', is_draw=False):
    """
    Input:
    patch_size:  the size of local patch
    num_nearest_neurons:  the num of nearest neurons to feed to PCA
    patch_root: root for npy files--img_stack.npy and ordered_neuron_position.npy

    Output:
    patches: in shape (num_frame, num_neuron, patch_size , patch_size, num_channel),
    by default it is (50, 107, 64, 64, 23)

    vertical_splices: (50, 107, 23, 23)
    """
    if not os.path.exists(file_root):
        os.makedirs(file_root)
    patch_file_path = os.path.join(file_root, 'rotated_patches_{}.npy'.format(patch_size))
    slice_file_path = os.path.join(file_root, 'vertical_slice.npy')
    if os.path.exists(patch_file_path):
        print('Patch file exits and it will be loaded')
        patches = np.load(patch_file_path)
        vertical_splices = np.load(slice_file_path)
        return patches, vertical_splices
    else:
        print('Get patch')

        num_frame, num_neuron = neuron_position.shape[0], neuron_position.shape[1]
        num_channel = frames.shape[3]

        patches = np.zeros((num_frame, num_neuron, patch_size, patch_size, num_channel))     # (50, 107, patch_size, patch_size, 23)
        vertical_splices = np.zeros((num_frame, num_neuron, num_channel, num_channel))       # (50, 107, 23, 23)

        for i in range(num_frame):
            cur_frame_pos = neuron_position[i, :, 0:2]
            # tail2head = cur_frame_pos[tail_neuron, 0:2] - cur_frame_pos[head_neuron, 0:2]
            tail2head = tail_pos[i, 0:2] - head_pos[i, 0:2]
            dist = -2 * cur_frame_pos.dot(cur_frame_pos.T) +  np.sum(cur_frame_pos ** 2, axis=1)[:, np.newaxis] +\
                    np.sum(cur_frame_pos ** 2, axis=1)
            cur_frame = frames[i, :, :, :]
            print("It's frame {}".format(i))
            for j in range(num_neuron):
                nearest_neurons = np.argsort(dist[j, :])[:num_nearest_neurons]
                nearest_neuron_positions = neuron_position[i, nearest_neurons, 0:2]
                pca = PCA()
                pca.fit(nearest_neuron_positions)
                pc = pca.components_[0, :]
                if np.sum(tail2head * pc) > 0:
                    pc = -pc       # keep the direction of pc consistent with the direction of tail2head
                pos = np.array(neuron_position[i, j, :])
                pos = pos.astype(int)
                drawing_root = './show/patch_size_{}/neuron_{}'.format(patch_size, j)

                crop_patch, vertical_splice = rotation_crop(direction=pc, x_0=pos[0], y_0=pos[1], frame=cur_frame,
                                                            width=patch_size, channel=pos[2], frame_id=i,
                                                            tail=tail_pos[i, 0:2],
                                                            head=head_pos[i, 0:2],
                                                            drawing_root=drawing_root, is_draw=is_draw
                                           )

                patches[i, j, :, :, :] = crop_patch
                vertical_splices[i, j, :, :] = vertical_splice

        patches = patches.astype(np.float32)
        vertical_splices = vertical_splices.astype(np.float32)
        np.save(patch_file_path, patches)
        np.save(slice_file_path, vertical_splices)
        return patches, vertical_splices
