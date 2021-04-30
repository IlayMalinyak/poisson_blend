import scipy.sparse
import numpy as np
from scipy.sparse.linalg import spsolve
import create_mask
import cv2
from matplotlib import pyplot as plt
from os import path
import time
import pyamg


def laplacian_matrix(n, m, k):
    mat_D = scipy.sparse.lil_matrix((n, n))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(6)
    mat_D.setdiag(-1, 1)

    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()

    mat_A.setdiag(-1, 1* m)
    mat_A.setdiag(-1, -1 * m)
    return mat_A


def possion_blend(target, src, mask, offset):
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

    # determines the diagonals on the coefficient matrix
    positions = np.where(mask)
    # setting the positions to be in a flatted manner
    positions = (positions[0] * region_size[1]) + positions[1]

    # row and col size of coefficient matrix
    n = np.prod(region_size)

    main_diagonal = np.ones(n)
    main_diagonal[positions] = 6
    diagonals = [main_diagonal]
    diagonals_positions = [0]

    # creating the diagonals of the coefficient matrix
    for diagonal_pos in [-1, 1,-region_size[1], region_size[1], -region_size[2], region_size[2]]:
        in_bounds_indices = None
        if np.any(positions + diagonal_pos > n):
            in_bounds_indices = np.where(positions + diagonal_pos < n)[0]
        elif np.any(positions + diagonal_pos < 0):
            in_bounds_indices = np.where(positions + diagonal_pos >= 0)[0]
        in_bounds_positions = positions[in_bounds_indices]

        diagonal = np.zeros(n)
        diagonal[in_bounds_positions + diagonal_pos] = -1
        if diagonals_positions == -region_size[1] or diagonal_pos == region_size[1]:
            zero_ind = [i*region_size[1]*region_size[0] - j for i in range(1, (n - region_size[1]) // (region_size[1]*region_size[0]))
                       for j in range(region_size[1])]
            diagonal[zero_ind] = 0
        diagonals.append(diagonal)
        diagonals_positions.append(diagonal_pos)
    A = scipy.sparse.spdiags(diagonals, diagonals_positions, n, n, 'csr')

    P = pyamg.gallery.poisson(mask.shape)
    # P = laplacian_matrix(mask.shape[0], mask.shape[1])

    # inverted_img_mask = np.invert(img_mask.astype(np.bool)).flatten()
    positions_from_target = np.where(mask.flatten()==0)[0]

    # get subimages
    t = target[region_target[0]:region_target[2], region_target[1]:region_target[3], :region_size[2]]
    s = src[region_source[0]:region_source[2], region_source[1]:region_source[3], :region_size[2]]
    t = t.flatten()
    s = s.flatten()

    # create b
    b = P * s
    b[positions_from_target] = t[positions_from_target]

        # solve Ax = b
    x = scipy.sparse.linalg.spsolve(A, b)

    # assign x to target image
    x = np.reshape(x, region_size)
    x = np.clip(x, 0, 255)
    x = np.array(x, target.dtype)
    # plt.imshow(x)
    # plt.show()
    target[region_target[0]:region_target[2], region_target[1]:region_target[3], :region_size[2]] = x

    return target


def create_3d_mask(src, representative = -1):
    mask = np.zeros(src.shape)
    pd = create_mask.PolygonDrawer()
    if representative == -1:
        for i in range(src.shape[2]):
            pd = create_mask.PolygonDrawer()
            mask[:,:,i] = pd.run(src[:,:,i], "slice number %d" % i)
    else:
        rep = pd.run(src[:,:, representative], "representative slice - %d" % representative)
        mask = np.repeat(rep[..., None], src.shape[2], axis=2)
    return mask


if __name__ == "__main__":
    brain = np.load("obj_3d.npy")
    plt.imshow(brain[...,3], cmap='gray')
    plt.show()
    bervet = np.load("obj_3d_small.npy")
    mask = create_3d_mask(bervet, representative=3)
    np.save("mask_full", mask)
    mask = np.load("mask.npy")
    new_target = possion_blend(brain[:,:,:10], bervet[:,:,:5], mask[:,:,:5], (20,20))
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(new_target[...,3], cmap='gray')
    ax[1].imshow(brain[...,3], cmap='gray')
    plt.show()
