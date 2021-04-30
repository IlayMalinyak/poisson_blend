import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import exposure

FINAL_LINE_COLOR = (10000, 255, 255)
WORKING_LINE_COLOR = (3000, 127, 127)
DEFAULT_SIZE = 512

# ============================================================================

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
            h,w = copy_im.shape
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
