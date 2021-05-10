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
import cv2
import skimage.io as io
import skimage.filters as filters
from skimage.util import img_as_ubyte
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage, stats
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy import ndimage
# from colloidspy import crop, analyze, process


def load(path, conserve_memory=True):
    stack = io.ImageCollection(path, conserve_memory=conserve_memory)
    return img_as_ubyte(stack), stack.files


class cspy_stack(np.ndarray):
    def __new__(cls, input_array, filenames=None, cropped=None, crop_coords=None, binary_otsu=None, binary_loc=None,
                binary_hyst=None, cleaned=None, particles=None):
        obj = np.asarray(input_array).view(cls)
        obj.filenames = filenames
        obj.cropped = cropped
        obj.crop_coords = crop_coords
        obj.binary_otsu = binary_otsu
        obj.binary_loc = binary_loc
        obj.binary_hyst = binary_hyst
        obj.cleaned = cleaned
        obj.particles = particles
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.filenames = getattr(obj, 'filenames', None)
        self.cropped = getattr(obj, 'cropped', None)
        self.crop_coords = getattr(obj, 'crop_coords', None)
        self.binary_otsu = getattr(obj, 'binary_otsu', None)
        self.binary_loc = getattr(obj, 'binary_loc', None)
        self.binary_hyst = getattr(obj, 'binary_hyst', None)
        self.cleaned = getattr(obj, 'cleaned', None)
        self.particles = getattr(obj, 'particles', None)

    def imshow(self, cmap='gray', **kwargs):
        plt.figure()
        plt.imshow(self, cmap=cmap, **kwargs)
        plt.tight_layout()
        plt.show()

    def add_cropped(self):
        def line_select_callback(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            self.crop_coords = ((int(x1), int(y1)), (int(x2), int(y2)))
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

        if len(self.shape) == 3:
            img = self[0]
        elif len(self.shape) == 2:
            img = self
        else:
            raise Exception

        select_roi(img)

        # Crop all images in the stack
        # Numpy's axis convention is (y,x) or (row,col)
        tpleft = [self.crop_coords[0][1], self.crop_coords[0][0]]
        btmright = [self.crop_coords[1][1], self.crop_coords[1][0]]
        if len(self.shape) == 3:
            self.cropped = cspy_stack(
                [self[i][tpleft[0]:btmright[0], tpleft[1]:btmright[1]] for i in range(len(self))])
        elif len(self.shape) == 2:
            img_rois = cspy_stack(self[tpleft[0]:btmright[0], tpleft[1]:btmright[1]])
        else:
            raise Exception

    def add_otsu(self, nbins=256):
        if self.cropped is None:
            if len(self.shape) == 3:
                self.binary_otsu = []
                for i in range(len(self)):
                    otsu = filters.threshold_otsu(self[i], nbins=nbins)
                    self.binary_otsu.append(self[i] > otsu)
            elif len(self.shape) == 2:
                otsu = filters.threshold_otsu(self, nbins=nbins)
                self.binary_otsu = self > otsu
            else:
                raise Exception
        else:
            if len(self.shape) == 3:
                self.binary_otsu = []
                for i in range(len(self)):
                    otsu = filters.threshold_otsu(self.cropped[i], nbins=nbins)
                    self.binary_otsu.append(self.cropped[i] > otsu)
            elif len(self.shape) == 2:
                otsu = filters.threshold_otsu(self.cropped, nbins=nbins)
                self.binary_otsu = self.cropped > otsu
            else:
                raise Exception

    def add_local_threshold(self, block_size=71, offset=5, cutoff=0, **kwargs):
        if self.cropped is None:
            if len(self.shape) == 3:
                self.binary_loc = []
                for i in range(len(self)):
                    local_thresh = filters.threshold_local(self[i], block_size=block_size, offset=offset, **kwargs)
                    low_val_flags = local_thresh < cutoff
                    local_thresh[low_val_flags] = 255
                    self.binary_loc.append(self[i] > local_thresh)
            elif len(self.shape) == 2:
                local_thresh = filters.threshold_local(self, block_size=block_size, offset=offset, **kwargs)
                low_val_flags = local_thresh < cutoff
                local_thresh[low_val_flags] = 255
                self.binary_loc = self > local_thresh
            else:
                raise Exception
        else:
            if len(self.shape) == 3:
                self.binary_loc = []
                for i in range(len(self)):
                    local_thresh = filters.threshold_local(self.cropped[i], block_size=block_size, offset=offset, **kwargs)
                    low_val_flags = local_thresh < cutoff
                    local_thresh[low_val_flags] = 255
                    self.binary_loc.append(self.cropped[i] > local_thresh)
            elif len(self.shape) == 2:
                local_thresh = filters.threshold_local(self.cropped, block_size=block_size, offset=offset, **kwargs)
                low_val_flags = local_thresh < cutoff
                local_thresh[low_val_flags] = 255
                self.binary_loc = self.cropped > local_thresh
            else:
                raise Exception

    def add_hysteresis_threshold(self, low=20, high=150):
        if self.cropped is None:
            if len(self.shape) == 3:
                self.binary_hyst = []
                for i in range(len(self)):
                    self.binary_hyst.append(filters.apply_hysteresis_threshold(self[i], low=low, high=high))
            elif len(self.shape) == 2:
                self.binary_hyst = filters.apply_hysteresis_threshold(self, low=low, high=high)
            else:
                raise Exception
        else:
            if len(self.shape) == 3:
                self.binary_hyst = []
                for i in range(len(self)):
                    self.binary_hyst.append(filters.apply_hysteresis_threshold(self.cropped[i], low=low, high=high))
            elif len(self.shape) == 2:
                self.binary_hyst = filters.apply_hysteresis_threshold(self.cropped, low=low, high=high)
            else:
                raise Exception

    def add_cleaned(self, bin_stack):
        if len(self.shape) == 3:
            self.cleaned = []
            for i in range(len(self)):
                self.cleaned.append(cspy_stack(ndimage.binary_closing(ndimage.binary_opening(bin_stack[i]))))
        elif len(self.shape) == 2:
            self.cleaned = cspy_stack(ndimage.binary_closing(ndimage.binary_opening(bin_stack)))
        else:
            raise Exception

    def find_particles(self, bin_stack, min_distance=7):
        if type(bin_stack) == list or len(bin_stack.shape) == 3:
            self.particles = []
            for i in range(len(bin_stack)):
                D = ndimage.distance_transform_edt(bin_stack[i])
                localMax = peak_local_max(D, indices=False, min_distance=min_distance, labels=bin_stack[i])
                markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
                labels = cspy_stack(watershed(-D, markers, mask=bin_stack[i]))
                self.particles.append(labels)
        elif len(bin_stack.shape) == 2:
            D = ndimage.distance_transform_edt(bin_stack)
            localMax = peak_local_max(D, indices=False, min_distance=min_distance, labels=bin_stack)
            markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
            labels = watershed(-D, markers, mask=bin_stack)
            self.particles = cspy_stack(labels)
        else:
            raise Exception

    def view_particles(self, img, particles, min_area=0, fill=False):
        if 'bool' in str(type(img[0][0])):
            clusters = cv2.cvtColor(np.zeros(img.shape, np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            clusters = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for particle in np.unique(particles):
            # if the label is zero, we are examining the 'background', so ignore it
            if particle == 0:
                continue
            # otherwise, allocate memory for the label region and draw it on the mask
            mask = np.zeros(img.shape, np.uint8)
            mask[particles == particle] = 255
            # detect contours in the mask and grab the largest one
            try:
                cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except ValueError:
                ct_im, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cv2.contourArea(cnts[0]) < min_area:
                continue
            if fill:
                cv2.drawContours(clusters, cnts, 0, (255, 255, 255), -1)
                cv2.drawContours(clusters, cnts, 0, (255, 0, 0), 0)
            else:
                cv2.drawContours(clusters, cnts, 0, (255, 0, 0), 0)
        return clusters

if __name__ == '__main__':
    os.chdir('C:/Users/Adam/OneDrive - University of Louisville/School/Masters Thesis/0.01 Temp6')
    stack = cspy_stack(*load('1 hour.tif'))
    stack.add_cropped()
    stack.add_local_threshold(block_size=31)
    stack.add_cleaned(stack.binary_loc)
    stack.find_particles(stack.cleaned, min_distance=3)
    io.imshow(stack.view_particles(stack.cropped[0], stack.particles[0]))



