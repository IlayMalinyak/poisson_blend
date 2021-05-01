import scipy.sparse
import numpy as np
from scipy.sparse.linalg import spsolve
import create_mask
from matplotlib import pyplot as plt
from utils import *
import pyamg
import possion2d


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
    region_size = (region_source[2] - region_source[0], region_source[3] - region_source[1], src.shape[2])
    mask = mask[region_source[0]:region_source[2], region_source[1]:region_source[3], :]  # todo check this
    mask[mask == 0] = False
    mask[mask != False] = True

    positions = np.where(mask.flatten())[0].astype(np.uint32)
    # positions = (positions[0] * region_size[1]) + positions[1]

    # n = number of element in region of blending col size of coefficient matrix
    n = np.prod(region_size).astype(np.uint64)

    main_diagonal = np.ones(n)
    main_diagonal[positions] = 6
    diagonals = [main_diagonal]
    diagonals_positions = [0]

    # creating the diagonals of the coefficient matrix
    for diagonal_pos in [-1, 1, -region_size[1], region_size[1], -region_size[1] * region_size[0],
                         region_size[1] * region_size[0]]:
        in_bounds_indices = None
        if np.any(positions + diagonal_pos >= n):
            in_bounds_indices = np.where(positions + diagonal_pos < n)[0]
        elif np.any(positions + diagonal_pos < 0):
            in_bounds_indices = np.where(positions + diagonal_pos >= 0)[0]
        in_bounds_positions = positions[in_bounds_indices]

        diagonal = np.zeros(n)
        diagonal[in_bounds_positions + diagonal_pos] = -1
        if abs(diagonal_pos) == region_size[1]:
            zero_ind = np.array([i * region_size[1] * region_size[0] - j for i in
                                 range(1, int(n / (region_size[1] * region_size[0])))
                                 for j in range(region_size[1], 0, -1)])
        elif abs(diagonal_pos) == region_size[1] * region_size[0]:  # todo maby redundant
            zero_ind = np.array([np.array([i * region_size[2] * region_size[1] * region_size[0] - j for i in
                                 range(1, int(n / (region_size[2] * region_size[1] * region_size[0])))
                                 for j in range(region_size[1] * region_size[0], 0, -1)])])
        else:
            zero_ind = np.arange(region_size[0] - 1, n - region_size[0], region_size[1])
        if len(zero_ind):
            in_bounds_zeros = np.arange(len(zero_ind))
            if np.any(zero_ind + diagonal_pos >= n):
                in_bounds_zeros = np.where(zero_ind + diagonal_pos < n)[0]
            elif np.any(zero_ind + diagonal_pos < 0):
                in_bounds_zeros = np.where(zero_ind + diagonal_pos >= 0)[0]
            if diagonal_pos > 0:
                diagonal[zero_ind[in_bounds_zeros].astype(np.int32) + diagonal_pos] = 0
            else:
                print(zero_ind, in_bounds_zeros)
                diagonal[zero_ind[in_bounds_zeros].astype(np.int32)] = 0

        diagonals.append(diagonal)
        diagonals_positions.append(diagonal_pos)
    P = laplacian_3d_matrix(region_size[0], region_size[1], region_size[2])
    A = scipy.sparse.spdiags(diagonals, diagonals_positions, n, n, 'csr')

    # inverted_img_mask = np.invert(img_mask.astype(np.bool)).flatten()
    positions_from_target = np.where(mask.flatten()==0)[0]

    x = solve(A, P, positions_from_target, region_size, region_source, region_target, src, target)
    target[region_target[0]:region_target[2], region_target[1]:region_target[3], :region_size[2]] = x

    return target


def run(target_path, src_path, mask_path=None, rep=-1):
    src = np.load(src_path)
    target = np.load(target_path)
    src += 1000
    if mask_path is None:
        mask = create_3d_mask(src, rep)
        display_axial(mask, None, None, 1)
    else:
        mask = np.load(mask_path)
    offset = get_offset(target[:,:,0], src[:,:,0])
    print(offset)
    if mask is not None:
        new_target = possion2d.poisson_blend(target[...,9], src[...,9], mask[...,9], offset)
        plt.imshow(target[...,9], cmap='gray')
        plt.show()
        plt.imshow(src[...,9], cmap='gray')
        plt.show()
        plt.imshow(new_target, cmap='gray')
        plt.show()
        # display_axial(mask, None, None, 1)


if __name__ == "__main__":
   run("obj_3d.npy", "obj_3d_small.npy", None, 3)