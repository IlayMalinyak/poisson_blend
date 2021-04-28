import scipy.sparse
import numpy as np
from scipy.sparse.linalg import spsolve
import create_mask
import cv2
from matplotlib import pyplot as plt
from os import path
import time
import pyamg


OFFSET_INSTRUCTIONS = "Mark the same blending position on both images and press Enter "


def laplacian_matrix(n, m):
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)

    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()

    mat_A.setdiag(-1, 1* m)
    mat_A.setdiag(-1, -1 * m)
    return mat_A


def generate_mask(source, fname):
    pd = create_mask.PolygonDrawer()
    mask = pd.run(source, "create mask")
    if mask is not None:
        cv2.imwrite("%s.png" % fname, mask)
        return mask
    return None


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
    region_size = (region_source[2] - region_source[0], region_source[3] - region_source[1])
    mask = mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    mask[mask == 0] = False
    mask[mask != False] = True

    # determines the diagonals on the coefficient matrix
    positions = np.where(mask)
    # setting the positions to be in a flatted manner
    positions = (positions[0] * region_size[1]) + positions[1]

    # row and col size of coefficient matrix
    n = np.prod(region_size)

    main_diagonal = np.ones(n)
    main_diagonal[positions] = 4
    diagonals = [main_diagonal]
    diagonals_positions = [0]

    # creating the diagonals of the coefficient matrix
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
    # P = laplacian_matrix(mask.shape[0], mask.shape[1])

    # inverted_img_mask = np.invert(img_mask.astype(np.bool)).flatten()
    positions_from_target = np.where(mask.flatten()==0)[0]

    for num_layer in range(target.shape[2]):
        # get subimages
        t = target[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer]
        s = src[region_source[0]:region_source[2], region_source[1]:region_source[3], num_layer]
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
        target[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer] = x

    return target


def get_offset(target, src):
    global posList
    posList = []

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            posList.append((x, y))
    image = target
    cv2.namedWindow(OFFSET_INSTRUCTIONS, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(OFFSET_INSTRUCTIONS, onMouse)
    while image is not None:
        cv2.imshow(OFFSET_INSTRUCTIONS, image)
        key = cv2.waitKey(0)
        # if the 'r' key is pressed, reset the cropping region
        if key == 13:
            image = src if image is target else None

    cv2.destroyAllWindows()
    return posList[1][0] - posList[0][0], posList[1][1] - posList[0][1]


def run(target_path,src_path, mask_path=None):
    src = cv2.imread(src_path)
    target = cv2.imread(target_path)
    if src.shape[0] > target.shape[0] or src.shape[1] > target.shape[1]:
        src = cv2.resize(src, (target.shape[1], target.shape[1]))
    if mask_path is None:
        mask = generate_mask(src, "mask")
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    offset = get_offset(target, src)
    print(offset)
    if mask is not None:
        new_target = possion_blend(target, src, mask, offset)
        # cv_clone = cv2.seamlessClone(src, target, mask, center_target, cv2.NORMAL_CLONE)
        # new_target = cv2.cvtColor(new_target, cv2.COLOR_BGR2RGB)
        plt.imshow(new_target[:, :, ::-1])
        plt.show()
        plt.imshow(src)
        plt.show()
        # plt.imshow(cv_clone[:,:,::-1])
        # plt.show()

if __name__ == "__main__":
    # scr_dir = 'fig'
    # source = cv2.imread(path.join(scr_dir, "test1_src.png"))
    # target = cv2.imread(path.join(scr_dir, "test1_target.png"))
    # mask = cv2.imread(path.join(scr_dir, "test1_mask.png"),
    #                   cv2.IMREAD_GRAYSCALE)
    # new_target = possion_blend(target, source, mask, (40,-30))
    # plt.imshow(new_target[:, :, ::-1])
    # plt.show()
    run("Neil-Young.jpg", "trump.jpeg")
