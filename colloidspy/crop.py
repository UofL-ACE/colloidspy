import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from skimage.util import img_as_ubyte


def crop_imgs(img_stack):
    """
    :param img_stack:
    :return: stack of cropped images as numpy array
    When rectangle has been selected, close the plot. Confirmed to work with PyQT5.
    """

    def line_select_callback(eclick, erelease):
        global x1, y1, x2, y2
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))

    def toggle_selector(event):
        print(' Key pressed.')
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            print(' RectangleSelector activated.')
            toggle_selector.RS.set_active(True)

    def select_roi(img):
        fig, current_ax = plt.subplots()
        plt.title("Select ROI, press any key to continue.")
        plt.imshow(img, cmap='gray')  # show the first image in the stack
        toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                               drawtype='box', useblit=True,
                                               button=[1, 3],  # don't use middle button
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)
        plt.connect('key_press_event', toggle_selector)
        plt.show()
        keyboardClick = False
        while keyboardClick != True:
            keyboardClick = plt.waitforbuttonpress()
        plt.close(fig)

    if type(img_stack) != io.collection.ImageCollection:
        try:
            img_stack = np.array(img_stack)
        except TypeError:
            print("Please pass a skimage image collection or a numpy array.")
            exit

    if type(img_stack) == io.collection.ImageCollection or len(img_stack.shape) == 3:
        img = img_as_ubyte(img_stack[0])
    elif len(img_stack.shape) == 2:
        img = img_as_ubyte(img_stack)
    else:
        raise TypeError

    select_roi(img)

    # Crop all images in the stack
    # Numpy's axis convention is (y,x) or (row,col)
    tpleft = [int(y1), int(x1)]
    btmright = [int(y2), int(x2)]
    if type(img_stack) == io.collection.ImageCollection or len(img_stack.shape) == 3:
        img_rois = np.asarray(
            [img_stack[i][tpleft[0]:btmright[0], tpleft[1]:btmright[1]] for i in range(len(img_stack))])
    elif len(img_stack.shape) == 2:
        img_rois = np.array(img_stack[tpleft[0]:btmright[0], tpleft[1]:btmright[1]])
    else:
        raise TypeError

    return img_rois


def cv_image_cropper(img_stack):
    """
    Alternate image cropped that uses OpenCV. May not work on Linux distros without the official opencv package.
    Select the region if interest (ROI) with your mouse.
    Press "Esc" to reselect ROI, press any other key to save current ROIs.
    :param img_stack: skimage image collection or numpy array.
    :return: numpy array of cropped images.
    """
    import cv2

    def on_mouse(event, x, y, flags, params):
        """
        Mouse callback for image cropper
        """
        global boxes
        if event == cv2.EVENT_LBUTTONDOWN:
            print('Start Mouse Position: ' + str(x) + ', ' + str(y))
            sbox = [x, y]
            boxes.append(sbox)

        elif event == cv2.EVENT_LBUTTONUP:
            print('End Mouse Position: ' + str(x) + ', ' + str(y))
            ebox = [x, y]
            boxes.append(ebox)
            print(boxes)
            crop = img[boxes[-2][1]:boxes[-1][1], boxes[-2][0]:boxes[-1][0]]
            cv2.imshow('crop', crop)
        k = cv2.waitKey(0)
        if k == 27:  # if "Esc" is pressed, reset crop
            cv2.destroyWindow('crop')
            boxes = []
            on_mouse(event, x, y, flags, params)
        else:
            cv2.destroyAllWindows()

    global boxes
    boxes = []
    if type(img_stack) == io.collection.ImageCollection or len(img_stack.shape) == 3:
        img = img_as_ubyte(img_stack[0])
    elif len(img_stack.shape) == 2:
        img = img_as_ubyte(img_stack)
    else:
        raise TypeError
    if img.shape[1] > 1080:
        x_win = 1080
        y_win = int((img.shape[0] / img.shape[1]) * 1080)
    else:
        x_win = img.shape[0]
        y_win = img.shape[1]
    cv2.namedWindow('top image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('top image', x_win, y_win)
    cv2.setMouseCallback('top image', on_mouse, 0)
    cv2.imshow('top image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Crop all images in the stack
    # OpenCV's axis convention is (x,y) instead of (y,x) or (row,col) in numpy
    tpleft = boxes[0]
    btmright = boxes[1]
    if type(img_stack) == io.collection.ImageCollection or len(img_stack.shape) == 3:
        img_rois = np.asarray(
            [img_stack[i][tpleft[1]:btmright[1], tpleft[0]:btmright[0]] for i in range(len(img_stack))])
    elif len(img_stack.shape) == 2:
        img_rois = np.array(img_stack[tpleft[1]:btmright[1], tpleft[0]:btmright[0]])
    else:
        raise TypeError

    return img_rois
