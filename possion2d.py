import scipy.sparse
import numpy as np
from scipy.sparse.linalg import spsolve
import create_mask
import cv2
from matplotlib import pyplot as plt
from utils import *
import pyamg


def poisson_blend(target, src, mask, offset):
    region_source = (
        max(-offset[0], 0),
        max(-offset[1], 0),
        min(target.shape[0] - offset[0], src.shape[0]),
        min(target.shape[1] - offset[1], src.shape[1]))
    region_target = (
        max(offset[0], 0),
        max(offset[1], 0),
        min(target.shape[0], src.shape[0] + offset[0]),
        min(target.shape[1], src.shape[1] + offset[1]))
    region_size = (region_source[2] - region_source[0], region_source[3] - region_source[1])
    mask = mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    mask[mask == 0] = False
    mask[mask != False] = True

    positions = np.where(mask)
    positions = (positions[0] * region_size[1]) + positions[1]
    # positions2 = np.where(mask.flatten())[0]

    # n = number of element in region of blending
    n = np.prod(region_size)

    main_diagonal = np.ones(n)
    main_diagonal[positions] = 4
    diagonals = [main_diagonal]
    diagonals_positions = [0]

    for diagonal_pos in [-1, 1, -region_size[1], region_size[1]]:
        in_bounds_indices = None
        if np.any(positions + diagonal_pos > n):
            in_bounds_indices = np.where(positions + diagonal_pos < n)[0]
        elif np.any(positions + diagonal_pos < 0):
            in_bounds_indices = np.where(positions + diagonal_pos >= 0)[0]
        in_bounds_positions = positions[in_bounds_indices]

        diagonal = np.zeros(n)
        diagonal[in_bounds_positions + diagonal_pos] = -1
        diagonals.append(diagonal)
        diagonals_positions.append(diagonal_pos)
    A = scipy.sparse.spdiags(diagonals, diagonals_positions, n, n, 'csr')
    P = pyamg.gallery.poisson(mask.shape)

    positions_from_target = np.where(mask.flatten() == 0)[0]
    if len(target.shape) > 2:
        for channel in range(target.shape[2]):
            x = solve(A, P, positions_from_target, region_size, region_source, region_target, src, target, channel)
            target[region_target[0]:region_target[2], region_target[1]:region_target[3], channel] = x
    else:
        x = solve(A, P, positions_from_target, region_size, region_source, region_target, src, target)
        target[region_target[0]:region_target[2], region_target[1]:region_target[3]] = x
    return target


def run(target_path,src_path, mask_path=None):
    src = cv2.imread(src_path)
    target = cv2.imread(target_path)
    plt.imshow(target)
    plt.show()
    plt.imshow(src)
    plt.show()
    # if src.shape[0] > target.shape[0] or src.shape[1] > target.shape[1]:
    #     src = cv2.resize(src, (target.shape[1], target.shape[1]))
    if mask_path is None:
        mask = create_2d_mask(src, "mask")
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    offset = get_offset(target, src)
    print(offset)
    if mask is not None:
        new_target = poisson_blend(target, src, mask, offset)
        # cv_clone = cv2.seamlessClone(src, target, mask, center_target, cv2.NORMAL_CLONE)
        # new_target = cv2.cvtColor(new_target, cv2.COLOR_BGR2RGB)
        plt.imshow(new_target[:, :, ::-1])
        plt.show()
        plt.imshow(src)
        plt.show()
        # plt.imshow(cv_clone[:,:,::-1])
        # plt.show()


if __name__ == "__main__":
    run("boddha.jfif", "lena.png")
