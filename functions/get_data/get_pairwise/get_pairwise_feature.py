from sklearn.decomposition import PCA
import numpy as np
from functions.get_data.get_pairwise.compute_sphere import compute_sphere

SCALE = 1.617

def get_pairwise_feature(cur_frame_position, num_edges, num_bins, tail2head, num_nearest_neurons=50, sphere_dim=3, origin_sphere_dim=5):
    """
    Input:
    cur_position: neuron position for one single frame , shape (107, 3)
    num_edge: num of connect edges for one neuron
    num_bins: to divide 2*pi into how many parts

    Output:
    hist_pairwise_feature: (num_neurons, dim), dim = 3 * num_bins
    sphere_feature_matrix: (num_neurons, dim') dim' = 5 * num_edges
    """
    cur_frame_position[:, 2] = cur_frame_position[:, 2] * SCALE
    pairwise_dim = sphere_dim * num_bins
    normal_pairwise_dim = num_edges * origin_sphere_dim
    num_neurons = cur_frame_position.shape[0]
    bin_angle = 2 * np.pi / num_bins

    dist = -2 * cur_frame_position.dot(cur_frame_position.T) + np.sum(cur_frame_position ** 2, axis=1)[:, np.newaxis] + \
           np.sum(cur_frame_position ** 2, axis=1)

    nearest_neurons = np.argsort(dist, axis=1)[:, 1: num_edges + 1]  # nearest neurons of one neuron except itself
    nearest_neurons_pc = np.argsort(dist, axis=1)[:, 0: num_nearest_neurons]          # for choose pc
    hist_pairwise_feature = np.zeros((num_neurons, pairwise_dim))
    sphere_feature_matrix = np.zeros((num_neurons, normal_pairwise_dim))
    for neuron in range(num_neurons):
        cur_neuron_position = cur_frame_position[neuron, :]
        nearest_neuron_position = cur_frame_position[nearest_neurons[neuron], :]
        nearest_neuron_position_pc = cur_frame_position[nearest_neurons_pc[neuron], :]
        sphere_feature = compute_sphere(cur_neuron_position, nearest_neuron_position)  # (num_edges, 5)

        sphere_feature_matrix[neuron, :] = sphere_feature.flatten()

        pca = PCA()
        pca.fit(nearest_neuron_position_pc[:, :2])
        pc = pca.components_[0, :]
        if np.sum(tail2head * pc) > 0:
            pc = -pc
        bins = [[] for _ in range(num_bins)]
        hist_features = np.zeros((num_bins, sphere_dim))

        edge_vec = cur_neuron_position - nearest_neuron_position
        for i in range(num_edges):
            cos_theta = np.sum(pc * edge_vec[i][:2]) / np.sqrt(np.sum(edge_vec[i][:2] ** 2))
            theta = np.arccos(cos_theta)
            cross_pc = pc[0] * edge_vec[i, 1] - pc[1] * edge_vec[i, 0] # https://blog.csdn.net/DinnerHowe/article/details/80923366
            if cross_pc > 0:
                theta = 2 * np.pi - theta
            bin_idx = int(theta // bin_angle)

            bins[bin_idx].append(sphere_feature[i, :3])  # without phi feature

        for i in range(num_bins):
            if len(bins[i]) > 0:
                sphere_feature_in_bin = np.array(bins[i]).reshape(-1, sphere_dim)
                sphere_feature_in_bin = np.mean(sphere_feature_in_bin, axis=0)
                hist_features[i, :] = sphere_feature_in_bin
        hist_features = np.reshape(hist_features, -1)
        hist_pairwise_feature[neuron, :] = hist_features

    return hist_pairwise_feature.astype(np.float32), sphere_feature_matrix.astype(np.float32)