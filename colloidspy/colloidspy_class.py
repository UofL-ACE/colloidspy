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
import matplotlib.pyplot as plt
from colloidspy import crop, analyze, process


def load(path, conserve_memory=True):
    stack = io.ImageCollection(path, conserve_memory=conserve_memory)
    return stack, stack.files


class cspy_stack(np.ndarray):
    def __new__(cls, input_array, filenames=None):
        obj = np.asarray(input_array).view(cls)
        obj.filenames = filenames
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    def imshow(self, **kwargs):
        plt.figure()
        plt.imshow(self, **kwargs)
        plt.show()

    def as_binary(self, bin_method='otsu'):
        if bin_method == 'otsu':
            process.otsu_threshold(self, nbins=256)
        elif bin_method == 'local':
            process.loc_threshold(self, block_size=71, offset=5, cutoff=0, method='gaussian')
        elif bin_method == 'hysteresis':
            process.hysteresis_theshold(self, low=20, high=150)
        else:
            print('Please provide a valid method:\n otsu, local, hysteresis')


if __name__ == '__main__':
    os.chdir('C:/Users/Adam/OneDrive - University of Louisville/School/Masters Thesis/0.01 Temp6')
    stack = cspy_stack(*load('*.tif'))
    # stack[0].imshow(cmap='gray')

