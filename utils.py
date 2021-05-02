import cv2
import scipy
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure


OFFSET_INSTRUCTIONS = "Mark the same blending position on both images and press Enter "
FINAL_LINE_COLOR = (10000, 255, 255)
WORKING_LINE_COLOR = (3000, 127, 127)
DEFAULT_SIZE = 512


def get_offset(target, src):
    """
    generate offset between target and source image
    :param target: target image
    :param src: source image
    :return: center of ROI offset - target.x - source.x, target.y - source.y
    """
    global posList
    posList = []

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            posList.append((x, y))
    image = exposure.rescale_intensity(target)
    current = "target"
    cv2.namedWindow(OFFSET_INSTRUCTIONS, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(OFFSET_INSTRUCTIONS, onMouse)
    while image is not None:
        cv2.imshow(OFFSET_INSTRUCTIONS, image)
        key = cv2.waitKey(0)
        # ord(enter) is 13
        if key == 13:
            image = exposure.rescale_intensity(src) if current == "target" else None
            current = "source" if current == "target" else None

    cv2.destroyAllWindows()
    return posList[0][1] - posList[1][1], posList[0][0] - posList[1][0]


def laplacian_matrix(n, m):
    """
    create 2d laplacian matrix
    :param n: rows number
    :param m: columns number
    :return: 2d laplacian matrix with shape (n*m, n*m)
    """
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)

    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()

    mat_A.setdiag(-1, 1* m)
    mat_A.setdiag(-1, -1 * m)
    return mat_A


def laplacian_3d_matrix(n,m,k):
    """
    create 3d laplacian matrix
    :param n: row number
    :param m: column number
    :param k: depth number
    :return: laplacian matrix with shape (n*m*k, n*m*k)
    """
    N = n*m*k
    main_diagonal = np.ones(N) * 6
    diagonals = [main_diagonal]
    diagonals_positions = [0]
    for diagonal_pos in [-1, 1, -m, m, -m*n,
                         m*n]:
        diagonal = np.ones(N) * -1
        if abs(diagonal_pos) == m:
            zero_ind = np.array([i *m * n - j for i in
                                 range(1, N // (m * n))
                                 for j in range(m, 0, -1)])
        elif abs(diagonal_pos) == 1:
            zero_ind = np.arange(n - 1, N - n, m)

        else:
            zero_ind = np.array([])
        if len(zero_ind):
            if diagonal_pos > 0:
                diagonal[zero_ind + diagonal_pos] = 0
            else:
                diagonal[zero_ind] = 0
        diagonals.append(diagonal)
        diagonals_positions.append(diagonal_pos)
    mat = scipy.sparse.spdiags(diagonals, diagonals_positions, N, N, 'csr')
    return mat


def flatten_3d_to_1d(arr):
    """
    flatt an array in the order (x,y,z)
    :param arr: 3d arr
    :return: 1d array where result[z*shape[x]*shape[y] + y*shape[x] + x] = original[x,y,z]
    """
    n = arr.shape[2]*arr.shape[1]*arr.shape[0]
    res = np.zeros(n)
    for i in range(arr.shape[2]):
        for j in range(arr.shape[1]):
            cur = i * arr.shape[1]*arr.shape[0] + j * arr.shape[1]
            res[cur: cur+arr.shape[1]] = arr[j,:,i]
    return np.array(res)


def reshape_1d_to_3d(arr, size):
    """
    reshape 1d array to 3d in (x,y,z) order
    :param arr: 1d array
    :param size: 3 elements tuple of new size
    :return: 3d array where result[x,y,z] = original[z*shape[x]*shape[y] + y*shape[x] + x]
    """
    res = np.zeros(size)
    for i in range(size[2]):
        print(i-1 * size[1] * size[0])
        s = arr[i * size[1] * size[0]:i * size[1] * size[0]]
        res[:,:,i] = arr[i * size[1] * size[0]:(i+1) * size[1] * size[0]].reshape((size[0], size[1]))
    return res


def solve(A, P, t,s, positions_from_target):
    """
    
    :param A:
    :param P:
    :param t:
    :param s:
    :param positions_from_target:
    :return:
    """
    b = P * s
    b[positions_from_target] = t[positions_from_target]
    x = scipy.sparse.linalg.spsolve(A, b)
    x[positions_from_target] = t[positions_from_target]
    # x = np.reshape(x, region_size)
    # x = np.clip(x, 0, 255)
    # x = np.array(x, target.dtype)
    return x


def create_2d_mask(source, fname):
    pd = PolygonDrawer()
    mask = pd.run(source, "create mask")
    if mask is not None and fname:
        cv2.imwrite("%s.png" % fname, mask)
        return mask
    return None


def create_3d_mask(src, representative=-1):
    mask = np.zeros(src.shape)
    if representative == -1:
        for i in range(src.shape[2]):
            pd = PolygonDrawer()
            mask[:,:,i] = pd.run(src[:,:,i], "slice number %d" % i)
    else:
        pd = PolygonDrawer()
        rep = pd.run(src[:,:, representative], "representative slice - %d" % representative)
        mask = np.repeat(rep[..., None], src.shape[2], axis=2)
    return mask


class PolygonDrawer(object):
    def __init__(self):
        self.done = False  # Flag signalling we're done
        self.current = (0, 0)  # Current position, so we can draw the line-in-progress
        self.points = []  # List of points defining our polygon
        self.factor_x = 1
        self.factor_y = 1

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done:  # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            # print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            # print("Completing polygon with %d points." % len(self.points))
            self.done = True

    def run(self, image, window_name):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, image)
        cv2.waitKey(1)
        cv2.setMouseCallback(window_name, self.on_mouse)
        copy_im = image.copy()
        if copy_im.shape[0] < DEFAULT_SIZE:
            h,w = copy_im.shape[:2]
            self.factor_x,  self.factor_y = DEFAULT_SIZE / w, DEFAULT_SIZE / h
            copy_im = cv2.resize(copy_im, (int(w * self.factor_x), int(h * self.factor_y)))
        copy_im = exposure.rescale_intensity(copy_im)
        while (not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            # canvas = np.zeros(image.shape(), np.uint8)
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                # cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 1)
                cv2.polylines(copy_im, np.array([self.points]).astype(np.int32), False, FINAL_LINE_COLOR, 2)
                # And  also show what the current segment would look like
                cv2.line(copy_im, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(window_name, copy_im)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 13:  # Enter hit
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing
        canvas = np.zeros((image.shape[:2]), np.uint8)
        # of a filled polygon
        if (len(self.points) > 0):
            resized_points = np.array([self.points]) / np.array([self.factor_x, self.factor_y])
            cv2.fillPoly(canvas,resized_points.astype(np.int32) , FINAL_LINE_COLOR)
        # And show it
        canvas = cv2.GaussianBlur(canvas, (5, 5), 0)
        _, canvas = cv2.threshold(canvas, 1, 255, cv2.THRESH_BINARY)
        cv2.imshow(window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()
        cv2.destroyWindow(window_name)
        return canvas


class IndexTracker(object):
    def __init__(self, ax, X, contours=None, seeds=None, aspect=1):
        self.ax = ax
        ax.set_title('Scroll to Navigate through the DICOM Image Slices')

        self.X = X
        self.contours = contours
        self.seeds = seeds
        if len(X.shape) == 3:
            rows, cols, self.slices = X.shape
            self.channels = 0
        else:
            rows, cols, self.channels, self.slices = X.shape
        self.ind = self.slices//2
        self.im = ax.imshow(self.X[..., self.ind], cmap='gray')
        self.struct = ax.imshow(self.contours[..., self.ind], cmap='cool', interpolation='none',
                                alpha=0.7) if self.contours is not None else None
        # draw_seeds_mask(ax, X, seeds_position)
        self.plan = ax.imshow(self.seeds[..., self.ind]) if self.seeds is not None else None
        ax.set_aspect(aspect)
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        print(event.key)
        if event.button == 'up':
            self.ind = (self.ind - 1) % self.slices
        else:
            self.ind = (self.ind + 1) % self.slices
        self.update()

    def on_key(self, event):
        print(event.key)
        if event.key == 'up':
            self.ind = (self.ind - 1) % self.slices
        else:
            self.ind = (self.ind + 1) % self.slices
        self.update()


    def update(self):
        self.im.set_data(self.X[..., self.ind])
        if self.contours is not None:
            self.struct.set_data(self.contours[:,:, self.ind])
        if self.seeds is not None:
            self.plan.set_data(self.seeds[:,:, self.ind])
        self.ax.set_ylabel('Slice Number: %s' % self.ind)
        # cid = self.ax.canvas.mpl_connect('key_press_event', self.on_key)
        self.im.axes.figure.canvas.draw()


def display_axial(arr, contours, seeds, aspect, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    masked_contour_arr = np.ma.masked_where(contours == 0, contours) if contours is not None else None
    # masked_seed_arr = np.ma.masked_where(seeds == 255, seeds) if seeds is not None else None
    tracker = IndexTracker(ax, arr, masked_contour_arr, seeds, aspect)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('key_press_event', tracker.on_key)

    plt.show()