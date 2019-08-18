import numpy as np


def cal_rotated_box_pos(pc_vec, center, half_width):
    vertical_vec = np.array([-pc_vec[1], pc_vec[0]])
    if pc_vec[0]*vertical_vec[1] - pc_vec[1]*vertical_vec[0] > 0:  # if vertical_vec is at the clock-wise of pc, we will reverse it direction
        vertical_vec = -vertical_vec
    cnt = np.zeros((4, 2))

    cnt[0, :] = center + half_width*pc_vec + half_width*vertical_vec
    cnt[1, :] = center + half_width*pc_vec - half_width*vertical_vec
    cnt[2, :] = center - half_width*pc_vec - half_width*vertical_vec
    cnt[3, :] = center - half_width*pc_vec + half_width*vertical_vec

    return cnt


