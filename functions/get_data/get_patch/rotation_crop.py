import cv2
import os
import numpy as np
from .cal_rotated_box_pos import  cal_rotated_box_pos

def rotation_crop(direction, x_0, y_0, frame, width, channel,frame_id,  tail, head, drawing_root, is_draw=False):
    """
    This function rotate the image and crop it to get local patch and vertical slice
    """
    img = frame   # (1024, 1024, 23)
    num_channel = img.shape[2]

    # get the rotated box coordinate, mainly for visualization
    cnt = cal_rotated_box_pos(direction, np.array([x_0, y_0]), int(width // 2))
    cnt = cnt.astype(int)
    rect = cv2.minAreaRect(cnt)

    # calculate the rotation angel
    if direction[0] > 0 and direction[1] < 0:
        angle = np.rad2deg(np.arctan(np.abs(direction[0] / direction[1])))
    if direction[0] < 0 and direction[1] < 0:
        angle = -np.rad2deg(np.arctan(np.abs(direction[0] / direction[1])))
    if direction[0] < 0 and direction[1] > 0:
        angle = -(180 - np.rad2deg(np.arctan(np.abs(direction[0] / direction[1]))))
    if direction[0] > 0 and direction[1] > 0:
        angle = (180 - np.rad2deg(np.arctan(np.abs(direction[0] / direction[1]))))

    img_crop, img_rot = crop_rect(img, rect, width, angle)
    vertical_slice_filling = np.zeros((num_channel, 2*num_channel))
    vertical_slice_filling[:, (num_channel - channel): (2*num_channel - channel)] = \
        img_rot[y_0, int(x_0 - num_channel//2): int(x_0 + num_channel//2 + 1), :]

    vertical_slice = vertical_slice_filling[:, num_channel - int(num_channel//2): num_channel + int(num_channel//2) + 1]

    if is_draw:
        if not os.path.exists(drawing_root):
            os.makedirs(drawing_root)
        box_rot_save_path = os.path.join(drawing_root, 'frame_{:02d}_box_rot.jpg'.format(frame_id))
        img_rot_save_path = os.path.join(drawing_root, 'frame_{:02d}_img_rot.jpg'.format(frame_id))
        slice_path = os.path.join(drawing_root, 'frame_{:02d}_slice_channel_{}.jpg'.format(frame_id, channel))
        # drawing the bndbox
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        print("bounding box: {}".format(box))
        drawing_frame = (frame[:, :, channel].copy() / 2).astype(np.uint8)
        cv2.drawContours(drawing_frame , [box], 0, (0, 0, 255), 2)
        cv2.line(drawing_frame, (int(x_0), int(y_0)), (int(x_0 + 40 * direction[0]), int(y_0 + 40 * direction[1])), (0, 0, 255), 1)
        cv2.line(drawing_frame, (int(tail[0]), int(tail[1])), (int(head[0]), int(head[1])), (0, 0, 255), 2)
        cv2.imwrite(box_rot_save_path, drawing_frame)
        # img_crop will the cropped rectangle, img_rot is the rotated image

        draw = img_rot[:, :, channel].copy()
        cv2.line(draw, (int(x_0 - width/2), int(y_0 - width/2)), (int(x_0 + width/2), int(y_0 - width/2)),
                 (0, 0, 255), 1)
        cv2.line(draw, (int(x_0 + width / 2), int(y_0 - width / 2)), (int(x_0 + width / 2), int(y_0 + width / 2)),
                 (0, 0, 255), 1)
        cv2.line(draw, (int(x_0 - width / 2), int(y_0 + width / 2)), (int(x_0 + width / 2), int(y_0 + width / 2)),
                 (0, 0, 255), 1)
        cv2.line(draw, (int(x_0 - width / 2), int(y_0 - width / 2)), (int(x_0 - width / 2), int(y_0 + width / 2)),
                 (0, 0, 255), 1)
        cv2.imwrite(img_rot_save_path, draw)

        cv2.imwrite(slice_path, vertical_slice)
    return img_crop, vertical_slice


def crop_rect(img, rect, patch_width, angle):
    num_channel = img.shape[2]
    center, size = rect[0], rect[1]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    height, width = img.shape[0], img.shape[1]

    # get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical and we crop it
    img_rot = (img_rot / 2).astype(np.uint8)    # from uint16 to uint8
    patches = np.zeros((patch_width, patch_width, num_channel))
    for i in range(img_rot.shape[2]):
        img_crop = cv2.getRectSubPix(img_rot[:,:,i], size, center)
        img_crop = cv2.resize(img_crop, (patch_width, patch_width))
        patches[:, :, i] = img_crop
    return patches, img_rot

