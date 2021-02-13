import numpy as np


def get_bbox_overlap_ratio(box1, all_boxed):
    """
    Info: https://stackoverflow.com/a/49301979/11180198
    :param box1: np.array, (xmin,ymin,boxwidth,boxheight)
    FIXME: optimization
    """
    # when the boxes are the same

    box1_left_x = box1[0]
    box2_left_x = all_boxed[0]
    box1_right_x = box1[0] + box1[2]
    box2_right_x = all_boxed[0] + all_boxed[2]
    box1_top_y = box1[1]
    box2_top_y = all_boxed[1]
    box1_bottom_y = box1[1] + box1[3]
    box2_bottom_y = all_boxed[1] + all_boxed[3]

    box1_area = box1[2] * box1[3]

    # situation when there is overlapping
    mask1 = box1_right_x > box2_left_x
    mask2 = box1_left_x < box2_right_x
    mask3 = box1_bottom_y > box2_top_y
    mask4 = box1_top_y < box2_bottom_y
    mask = np.bitwise_and(np.bitwise_and(mask1, mask2), np.bitwise_and(mask3, mask4))

    overlap_left_x = np.maximum(box1_left_x, box2_left_x)
    overlap_right_x = np.minimum(box1_right_x, box2_right_x)
    overlap_top_y = np.maximum(box1_top_y, box2_top_y)
    overlap_bottom_y = np.minimum(box1_bottom_y, box2_bottom_y)
    overlap_area = (overlap_right_x - overlap_left_x) * (overlap_bottom_y - overlap_top_y)

    overlap_ratios = overlap_area / box1_area
    return np.where(mask, overlap_ratios, np.zeros_like(overlap_ratios))