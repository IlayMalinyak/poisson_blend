import scipy.sparse
import numpy as np
from scipy.sparse.linalg import spsolve
import create_mask
from matplotlib import pyplot as plt
from utils import *
import pyamg
import possion2d
import cv2


def poisson_blend(target, src, mask, offset):
    """
    implement poisson blending in 3d
    :param target: target image (in which we blend)
    :param src: source image (from which we take the blending object)
    :param mask: mask of the ROI - region for blending
    :param offset: offset between target and
    :return: target image with the object from source blended inside ROI
    """
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
    mask = mask[region_source[0]:region_source[2], region_source[1]:region_source[3], :]
    mask[mask == 0] = False
    mask[mask != False] = True

    # find the ROI (region in which we solve poisson eq.) because numpy flattening method is column major (flatten z,x,y)
    # and the laplacian operator matrix is flatten (x,y,z) we need to swap the axes befoer flattening
    positions = np.where(np.transpose(mask, (2,0,1)).flatten())[0]

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
        elif abs(diagonal_pos) == 1:
            zero_ind = np.arange(region_size[0] - 1, n - region_size[0], region_size[1])
        else:
            zero_ind = np.array([])
        if len(zero_ind):
            in_bounds_zeros = np.arange(len(zero_ind))
            if np.any(zero_ind + diagonal_pos >= n):
                in_bounds_zeros = np.where(zero_ind + diagonal_pos < n)[0]
            elif np.any(zero_ind + diagonal_pos < 0):
                in_bounds_zeros = np.where(zero_ind + diagonal_pos >= 0)[0]
            diagonal[zero_ind[in_bounds_zeros].astype(np.int32) + diagonal_pos] = 0

        diagonals.append(diagonal)
        diagonals_positions.append(diagonal_pos)
    P = laplacian_3d_matrix(region_size[0], region_size[1], region_size[2])  # pyamg is not good here
    A = scipy.sparse.spdiags(diagonals, diagonals_positions, n, n, 'csr')

    # find indices of target/ROI region
    positions_from_target = np.where(np.transpose(mask, (2,0,1)).flatten() == 0)[0]

    t = target[region_target[0]:region_target[2], region_target[1]:region_target[3], :]
    s = src[region_source[0]:region_source[2], region_source[1]:region_source[3], :]
    t = np.transpose(t, (2,0,1)).flatten()
    s = np.transpose(s, (2,0,1)).flatten()
    x = solve(A, P, t, s, positions_from_target, region_size)
    x = np.array(x, target.dtype)
    x = reshape_1d_to_3d(x, mask.shape)
    target[region_target[0]:region_target[2], region_target[1]:region_target[3], :region_size[2]] = x

    return target


def run(target_path, src_path, mask_path=None, rep=-1):
    src = np.load(src_path)
    target = np.load(target_path)
    # src += 1000
    if mask_path is None:
        # mask =[]
        mask = create_3d_mask(src, rep)
        np.save( "mask_3d", mask)
        display_axial(mask, None, None, 1)
    else:
        mask = np.load(mask_path)
        # mask = cv2.imread(mask_path)
    # offset = get_offset(target[:,:,0], src[:,:,0])
    # print(offset)
    if mask is not None:
        # mask = np.ones(src.shape)
        # src = src[:10,:10, :]
        # mask = create_3d_mask(src, rep)
        # np.save("mask_3d_small", mask)
        new_target = poisson_blend(target[...,:30], src, mask, (80,75))
        # for slice in range(5):
        #     target[:,:,slice] = possion2d.poisson_blend(target[:,:,slice], src[...,5], mask[...,slice], offset)
        slice_src = cv2.resize(src[...,9], (512,512))
        slice_target = cv2.resize(target[...,9], (512,512))
        # slice_mask = create_2d_mask(slice_src, "mask_slice")
        # new_target = possion2d.poisson_blend(slice_target,  slice_src, slice_mask, (100,100))
        # new_target_2 = cv2.seamlessClone(slice_src, slice_target, slice_mask, (200,200), flags=cv2.NORMAL_CLONE)
        # plt.imshow(new_target, cmap='gray')
        # plt.show()
        # plt.imshow(src[...,9], cmap='gray')
        # plt.show()
        # plt.imshow(new_target_2[...,9], cmap='gray')
        # plt.title("cv2")
        # plt.show()
        display_axial(src, None, None, 1)
        display_axial(new_target, None, None, 1)


if __name__ == "__main__":
   run("obj_3d.npy", "obj_3d_small.npy", "mask_3d.npy", rep=9)