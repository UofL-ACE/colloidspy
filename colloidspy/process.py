import numpy as np
import skimage.filters as filters
import skimage.io as io
from scipy import ndimage


def loc_threshold(img_stack, block_size=71, offset=5, cutoff=0, method='gaussian'):
    """
    :param img_stack: skimage image collection or numpy array.
    :param block_size: local region in pixels in which the threshold is evaluated.
    :param offset: pixel value to subtract from the weighted mean pixel value of the localized block area.
    :param cutoff: minimum pixel value to be considered a particle. Prevents black space from appearing as clusters.
    :param method: method used to determine threshold. Identical to skimage threshold_local methods.
    :return: numpy array of binary images
    """

    try:
        binary_stack = []
        for i in range(len(img_stack)):
            local_thresh = filters.threshold_local(img_stack[i], block_size, offset=offset, method=method)
            low_val_flags = local_thresh < cutoff
            local_thresh[low_val_flags] = 255
            binary_img = img_stack[i] > local_thresh
            binary_stack.append(binary_img)
    except ValueError:
        # value error probably means user fed a single image instead of a stack
        local_thresh = filters.threshold_local(img_stack, block_size, offset=offset, method=method)
        low_val_flags = local_thresh < cutoff
        local_thresh[low_val_flags] = 255
        binary_stack = img_stack > local_thresh

    return np.array(binary_stack)


def otsu_threshold(img_stack, nbins=256):
    """
    Applies a global threshold to a stack of images. Suitable for even images with good particle definition.
    :param img_stack:
    :param nbins:
    :return:
    """
    try:
        binary_stack = []
        for i in range(len(img_stack)):
            otsu = filters.threshold_otsu(img_stack[i], nbins=nbins)
            binary_img = img_stack[i] > otsu
            binary_stack.append(binary_img)
    except ValueError:
        # value error probably means user fed a single image instead of a stack
        otsu = filters.threshold_otsu(img_stack, nbins=nbins)
        binary_stack = img_stack[i] > otsu

    return np.array(binary_stack)


def clean_imgs(stack):
    """
    Removes single-pixel particles and gritty imperfections from a stack of binary images.
    :param stack: numpy array of binary images
    :return: numpy array of cleaned binary images.
    """

    if type(stack) == io.collection.ImageCollection or len(stack.shape) == 3:
        clean_stack = []
        for i in range(len(stack)):
            open_img = ndimage.binary_opening(stack[i])
            clean_stack.append(ndimage.binary_closing(open_img))
    elif len(stack.shape) == 2:
        clean_stack = ndimage.binary_closing(ndimage.binary_opening(stack))
    else:
        raise TypeError
    return np.array(clean_stack)
