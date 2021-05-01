import numpy as np
import scipy
import pyamg
from utils import *

def create_a_and_p_3d(region_size, positions):
    n = np.prod(region_size)

    main_diagonal = np.ones(n)
    main_diagonal[positions] = 6
    diagonals = [main_diagonal]
    diagonals_positions = [0]

    # creating the diagonals of the coefficient matrix
    for diagonal_pos in [-1, 1, -region_size[1], region_size[1], -region_size[1]*region_size[0], region_size[1]*region_size[0]]:
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
                        range(1, n // (region_size[1] * region_size[0]))
                        for j in range(region_size[1], 0, -1)])
        elif abs(diagonal_pos) == region_size[1]*region_size[0]: # todo maby redundant
            zero_ind = np.array([i * region_size[2] * region_size[1] * region_size[0] - j for i in
                                 range(1, n // (region_size[2] * region_size[1] * region_size[0]))
                                 for j in range(region_size[1]*region_size[0], 0, -1)])
        else:
            zero_ind = np.arange(region_size[0] - 1, n - region_size[0], region_size[1])
        if len(zero_ind):
            in_bounds_zeros = np.arange(len(zero_ind))
            if np.any(zero_ind + diagonal_pos >= n):
                in_bounds_zeros = np.where(zero_ind + diagonal_pos < n)[0]
            elif np.any(zero_ind + diagonal_pos < 0):
                in_bounds_zeros = np.where(zero_ind + diagonal_pos >= 0)[0]
            if diagonal_pos > 0:
                diagonal[zero_ind[in_bounds_zeros] + diagonal_pos] = 0
            else:
                diagonal[zero_ind[in_bounds_zeros]] = 0

        diagonals.append(diagonal)
        diagonals_positions.append(diagonal_pos)
    A = scipy.sparse.spdiags(diagonals, diagonals_positions, n, n, 'csr')
    P = laplacian_3d_matrix(region_size[0], region_size[1], region_size[2])
    return A,P


def create_a_and_p_2d(region_size, positions):
    n = np.prod(region_size)

    main_diagonal = np.ones(n)
    main_diagonal[positions] = 4
    diagonals = [main_diagonal]
    diagonals_positions = [0]

    for diagonal_pos in [-1, 1, -region_size[1], region_size[1]]:
        in_bounds_indices = None
        if np.any(positions + diagonal_pos >= n):
            in_bounds_indices = np.where(positions + diagonal_pos < n)[0]
        elif np.any(positions + diagonal_pos < 0):
            in_bounds_indices = np.where(positions + diagonal_pos >= 0)[0]
        in_bounds_positions = positions[in_bounds_indices]

        diagonal = np.zeros(n)
        diagonal[in_bounds_positions + diagonal_pos] = -1
        diagonals.append(diagonal)
        diagonals_positions.append(diagonal_pos)
    A = scipy.sparse.spdiags(diagonals, diagonals_positions, n, n, 'csr')
    P = pyamg.gallery.poisson(region_size)
    return A,P

p = np.array(pyamg.gallery.poisson((3,4)).todense())
n,m,k = 3,4,6
positions = np.arange(2*2*3)
a,p = create_a_and_p_3d((2,2,3), positions)
a,p = np.array(a.todense()), np.array(p.todense())
print(a -p)
