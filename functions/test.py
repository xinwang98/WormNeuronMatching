import torch
import time
import numpy as np
import h5py
import os
from scipy.io import loadmat
from scipy.special import softmax
from functions.model import ClassifierNetwork
from functions.get_data.get_patch.get_patch import get_patch
from functions.get_data.get_pairwise.get_pairwise_feature import get_pairwise_feature

def test(args, model_save_dir, num_frame=50, num_neuron=107):
    net_path = os.path.join(model_save_dir,
                                 '{}_bins_{}_edges_{}_epochs_{}.pth.tar'.format(args.model, args.bin_num, args.edge_num,
                                                                                args.epochs))
    test_data_save_dir = './data/test_data'
    if not os.path.exists(test_data_save_dir):
        os.makedirs(test_data_save_dir)

    position_path = os.path.join(test_data_save_dir, 'neuron_position.npy')
    label_path = os.path.join(test_data_save_dir, 'labels.npy')

    if os.path.exists(position_path) and os.path.exists(label_path):
        positions = np.load(position_path)
        labels = np.load(label_path)
    else:
        index_and_position = loadmat('./data/mat_files/index_and_position.mat')
        neuron_index = index_and_position['neuron_index_data']
        neuron_position = index_and_position['neuron_position_data'][:, 0]

        labels = np.zeros((num_frame, num_neuron))
        positions = np.zeros((num_frame, num_neuron, 3))

        for f in range(num_frame):
            idx_list = neuron_index[f, 0][0, :] - 1
            pos_list = neuron_position[f]
            labels[f, :] = idx_list
            positions[f, :, :] = pos_list.T
        np.save(position_path, positions)
        np.save(label_path, labels)

    image_path = os.path.join(test_data_save_dir, 'images.npy')
    if os.path.exists(image_path):
        images = np.load(image_path)
    else:
        img_Stack = h5py.File('./data/mat_files/img_Stack.mat')
        img_stack = img_Stack['img_Stack'][0, :]
        img_stack_extract = []
        for i in range(img_stack.shape[0]):
            img_stack_extract.append(np.transpose(img_Stack[img_stack[i]]))
        images = np.array(img_stack_extract)

        np.save(image_path, images)

    row_idx = np.arange(num_frame)
    head_idx = np.argwhere(labels == args.head_neuron)[:, 1]
    tail_idx = np.argwhere(labels == args.tail_neuron)[:, 1]

    head_pos = positions[row_idx, head_idx, :]
    tail_pos = positions[row_idx, tail_idx, :]
    patches, slices = get_patch(frames=images, neuron_position=positions, head_pos=head_pos, tail_pos=tail_pos,
                                file_root='./data/test_data')
    patches = np.transpose(patches, (0, 1, 4, 2, 3))

    slices = slices[:, :, :, :, np.newaxis]
    slices = np.transpose(slices, (0, 1, 4, 2, 3))

    patches -= np.mean(patches, axis=(0, 1))
    slices -= np.mean(slices, axis=(0, 1))

    classifier = ClassifierNetwork(bin_num=args.bin_num, edge_num=args.edge_num).cuda(args.gpu)
    classifier.load_state_dict((torch.load(net_path)['state_dict']))
    classifier.eval()

    total_acc = 0
    for count, frame_id in enumerate(range(40, 50)):
        tail2head = tail_pos[frame_id, 0:2] - head_pos[frame_id, 0:2]
        pairwise_hist_features, sphere_feature = get_pairwise_feature(cur_frame_position=positions[frame_id].copy(),
                                                   num_bins=args.bin_num, num_edges=args.edge_num, tail2head=tail2head)
        pairwise4classifier = np.concatenate((pairwise_hist_features, sphere_feature), axis=1)
        with torch.no_grad():
            unknown_patches = torch.from_numpy(patches[frame_id]).cuda(args.gpu)
            unknown_slices = torch.from_numpy(slices[frame_id]).cuda(args.gpu)
            pairwise4classifier = torch.from_numpy(pairwise4classifier).cuda(args.gpu)

            output = classifier(unknown_patches, unknown_slices, pairwise4classifier)
            out = output.cpu().numpy()
            prob = softmax(out, axis=1)
            s0 = prob
            for k in range(args.bi_stochastic_iter):
                s1 = s0 * (1 / (np.dot(np.ones((num_neuron, num_neuron)), s0)))
                s2 = (1 / (np.dot(s1, np.ones((num_neuron, num_neuron))))) * s1
                s0 = s2
            pred = np.argmax(s0, axis=1)
            acc = np.sum(pred == labels[frame_id]) / num_neuron
            print('Acc of Frame {} is {:.2f}'.format(frame_id, acc))
            total_acc += acc
    print('Total acc is {:.2f}'.format(total_acc / (count + 1)))