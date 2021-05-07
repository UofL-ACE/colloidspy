"""
2021.05 Redesign
_______________________________________________________________________________________________________________________
Original colloidspy design was just a series of functions that were not tied together in any meaningful way.
These functions each had their own error handling and type checking to make sure it would run smoothly, however they
were not very structured.
This version redesigns the architecture to an object-oriented style.
- Adam
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as io
import skimage.filters as filters
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
from colloidspy import crop, analyze, process


def load(path, conserve_memory=True):
    stack = io.ImageCollection(path, conserve_memory=conserve_memory)
    return stack, stack.files

def crop(img_stack):
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

    img_stack = cspy_stack(img_as_ubyte(img_stack))

    if len(img_stack.shape) == 3:
        img = img_stack[0]
    elif len(img_stack.shape) == 2:
        img = img_stack
    else:
        raise TypeError

    select_roi(img)

    # Crop all images in the stack
    # Numpy's axis convention is (y,x) or (row,col)
    tpleft = [int(y1), int(x1)]
    btmright = [int(y2), int(x2)]
    if len(img_stack.shape) == 3:
        img_rois = cspy_stack(
            [img_stack[i][tpleft[0]:btmright[0], tpleft[1]:btmright[1]] for i in range(len(img_stack))])
    elif len(img_stack.shape) == 2:
        img_rois = cspy_stack(img_stack[tpleft[0]:btmright[0], tpleft[1]:btmright[1]])
    else:
        raise TypeError

    return img_rois


class cspy_stack(np.ndarray):
    def __new__(cls, input_array, filenames=None):
        obj = np.asarray(input_array).view(cls)
        obj.filenames = filenames
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.filenames = getattr(obj, 'filenames', None)

    def imshow(self, cmap='gray', **kwargs):
        plt.figure()
        plt.imshow(self, cmap=cmap, **kwargs)
        plt.tight_layout()
        plt.show()

    def otsu_thresh(self, nbins=256):
        if len(self.shape) == 3:
            for i in range(len(self)):
                otsu = filters.threshold_otsu(self[i], nbins=nbins,)
                self[i] = self[i] > otsu
        elif len(self.shape) == 2:
            otsu = filters.threshold_otsu(self, nbins=nbins)
            self = self > otsu
        else:
            raise Exception
        return cspy_stack(self)

    def local_threshold(self, block_size=71, offset=5, cutoff=0, **kwargs):
        if len(self.shape) == 3:
            for i in range(len(self)):
                local_thresh = filters.threshold_local(self[i], block_size=block_size, offset=offset, **kwargs)
                low_val_flags = local_thresh < cutoff
                local_thresh[low_val_flags] = 255
                self[i] = self[i] > local_thresh
        elif len(self.shape) == 2:
            local_thresh = filters.threshold_local(self, block_size=block_size, offset=offset, **kwargs)
            low_val_flags = local_thresh < cutoff
            local_thresh[low_val_flags] = 255
            self = self > local_thresh
        else:
            raise Exception
        return cspy_stack(self)

    def hysteresis_threshold(self, low=20, high=150):
        if len(self.shape) == 3:
            for i in range(len(self)):
                self[i] = filters.apply_hysteresis_threshold(self[i], low=low, high=high)
        elif len(self.shape) == 2:
            self = filters.apply_hysteresis_threshold(self, low=low, high=high)
        else:
            raise Exception
        return self

    def clean(self):
        from scipy import ndimage
        if len(self.shape) == 3:
            for i in range(len(self)):
                self[i] = ndimage.binary_closing(ndimage.binary_opening(self[i]))
        elif len(self.shape) == 2:
            self = ndimage.binary_closing(ndimage.binary_opening(self))
        else:
            raise Exception
        return self

if __name__ == '__main__':
    os.chdir('C:/Users/Adam/OneDrive - University of Louisville/School/Masters Thesis/0.01 Temp6')
    stack = cspy_stack(*load('*.tif'))
    cropped = crop(stack)
    bin_stack = cropped.local_threshold().copy()
    # bin_stack[0].imshow()
    # plt.title('original')
    # clean_stack = bin_stack.clean()
    # clean_stack[0].imshow()
    # plt.title('cleaned')

