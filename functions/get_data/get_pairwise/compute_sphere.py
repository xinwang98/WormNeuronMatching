import numpy as np

def compute_sphere(pos, connected_pos):
    """
    Input:
    pos: position for one neuron, (3, )
    connected_pos: (N, 3), N is edge num

    Output:
    sphere feature: (N, 5);  r, sin theta, cos theta, sin phi, cos phi
    """
    connected_num = connected_pos.shape[0]
    sphere_feature = np.zeros((connected_num, 5))

    connected_edge_vec = pos - connected_pos

    connected_dis = np.sqrt(np.sum(connected_edge_vec ** 2, axis=1))
    sphere_feature[:, 0] = connected_dis / 100  # normalized distance
    theta = np.arccos(connected_edge_vec[:, 2] / connected_dis)
    sphere_feature[:, 1] = np.sin(theta)  # sin theta
    sphere_feature[:, 2] = np.cos(theta)  # cos theta
    sphere_feature[:, 3] = connected_edge_vec[:, 0] / np.sqrt(
        (np.sum(connected_edge_vec[:, 0:2] ** 2, axis=1)))  # sin phi
    sphere_feature[:, 4] = connected_edge_vec[:, 1] / np.sqrt(
        (np.sum(connected_edge_vec[:, 0:2] ** 2, axis=1)))  # cos phi

    return sphere_feature