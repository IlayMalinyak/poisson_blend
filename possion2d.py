import scipy.sparse
from utils import *
import pyamg


def poisson_blend(target, src, mask, offset):
    """
    implement poisson blending in 2d
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
    region_size = (region_source[2] - region_source[0], region_source[3] - region_source[1])
    mask = mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    mask[mask == 0] = False
    mask[mask != False] = True

    # find ROI (region in which we solve poisson eq.)
    positions = np.where(mask)
    positions = (positions[0] * region_size[1]) + positions[1]

    # n = number of element in region of blending
    n = np.prod(region_size)

    # creating the diagonals of the coefficient matrix
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
    if len(target.shape) > 2:  # rgb image
        for channel in range(target.shape[2]):
            t = target[region_target[0]:region_target[2], region_target[1]:region_target[3], channel]
            s = src[region_source[0]:region_source[2], region_source[1]:region_source[3], channel]
            t = t.flatten()
            s = s.flatten()
            x = solve(A, P, t, s, positions_from_target)
            x = np.reshape(x, region_size)
            x = np.clip(x, 0, 255)
            target[region_target[0]:region_target[2], region_target[1]:region_target[3], channel] = x
    else:  # grayscale image
        t = target[region_target[0]:region_target[2], region_target[1]:region_target[3]]
        s = src[region_source[0]:region_source[2], region_source[1]:region_source[3]]
        t = t.flatten()
        s = s.flatten()
        x = solve(A, P, t, s, positions_from_target)
        x = np.reshape(x, region_size)
        x = np.clip(x, 0, 255)
        target[region_target[0]:region_target[2], region_target[1]:region_target[3]] = x
    return target


def run(target_path,src_path, mask_path=None):
    src = cv2.imread(src_path)
    target = cv2.imread(target_path)
    if mask_path is None:
        mask = create_2d_mask(src, "mask")
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    offset = get_offset(target, src)
    print(offset)
    if mask is not None:
        target = poisson_blend(target, src, mask, offset)
        plt.imshow(target[:, :, ::-1])
        plt.show()
        plt.imsave("results/blend_2d.png", target[:, :, ::-1])
        plt.imsave("results/mask_blend_2d.png", mask, cmap='gray')


if __name__ == "__main__":
    run("obama.jfif", "trump2.jfif", "mask.png")
