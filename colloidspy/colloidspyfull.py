# -*- coding: utf-8 -*-
"""
2D Colloids Image Analysis Program
Original file - new is separated into modules
Created by Adam Cecil, University of Louisville, 2021
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats, ndimage
import skimage.io as io
import skimage.filters as filters
from skimage.util import img_as_ubyte
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


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

    if type(img_stack) == io.collection.ImageCollection or len(img_stack.shape) == 3:
        img = img_as_ubyte(img_stack[0])
    elif len(img_stack.shape) == 2:
        img = img_as_ubyte(img_stack)
    else:
        raise TypeError

    fig, current_ax = plt.subplots()
    plt.imshow(img, cmap='gray')  # show the first image in the stack
    toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.show()

    # Crop all images in the stack
    # Numpy's axis convention is (y,x) or (row,col)
    tpleft = [int(y1), int(x1)]
    btmright = [int(y2), int(x2)]
    if type(img_stack) == io.collection.ImageCollection or len(img_stack.shape) == 3:
        img_rois = np.asarray(
            [img_stack[i][tpleft[0]:btmright[0], tpleft[1]:btmright[1]] for i in range(len(img_stack))])
    elif len(img_stack.shape) == 2:
        img_rois = img_stack[tpleft[0]:btmright[0], tpleft[1]:btmright[1]]
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
        img_rois = img_stack[tpleft[1]:btmright[1], tpleft[0]:btmright[0]]
    else:
        raise TypeError

    return img_rois


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

    return binary_stack


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

    return binary_stack


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
    return clean_stack


def img_watershed(img, min_distance=7):
    """
    :param img: single image
    :param min_distance: minimum distance between particle mass centers.
    :return: 2D list of regions designating each particle - "blobs"
    Each region can be accessed using np.unique(), where 0 is the background.
    Adapted from https://www.pyimagesearch.com/2015/11/02/watershed-opencv/
    """

    img = img_as_ubyte(img)
    D = ndimage.distance_transform_edt(img)
    localMax = peak_local_max(D, indices=False, min_distance=min_distance, labels=img)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=img)

    return labels


def view_clusters(img, blobs):
    """
    :param img: single image.
    :param blobs: blobs taken from img_watershed.
    :return: RGB image of clusters detected by img_watershed.
    """

    clusters = cv2.cvtColor(np.zeros(img.shape, np.uint8), cv2.COLOR_GRAY2RGB)
    for particle in np.unique(blobs):
        # if the label is zero, we are examining the 'background', so ignore it
        if particle == 0:
            continue
        # otherwise, allocate memory for the label region and draw it on the mask
        mask = np.zeros(img.shape, np.uint8)
        mask[blobs == particle] = 255
        # detect contours in the mask and grab the largest one
        try:
            cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            ct_im, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(clusters, cnts, 0, (255, 255, 255), -1)
        cv2.drawContours(clusters, cnts, 0, (255, 0, 0), 0)

    return clusters


def structure_factor(img):
    """
    Does NOT return true structure factor, but can be used to calculate it.
    :param img:
    :return: list: radial average of the square of the 2D fourier transform of the image over wavenumber.
    """

    # Take 2D FFT, shift it to center
    H, W = np.shape(img)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(img)
    fsquare = fshift ** 2

    def radial_profile(data, center):
        y, x = np.indices((data.shape))
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        r = r.astype(np.int)

        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        return radialprofile

    sfactor = radial_profile(np.abs(fsquare), (H / 2, W / 2))

    return sfactor


def savedfs(df, directory, fname):
    """
    Saves given dataframe as .csv
    :param df: pandas dataframe to be saved.
    :param directory: directory to save the dataframe.
    :param fname: filename for the dataframe.
    :return:
    """

    if str(fname)[:-4] != '.csv':
        fname = str(fname) + '.csv'
    df.to_csv(os.path.join(directory, fname))


def saveimg(img, directory, fname):
    """
    Saves an image to the given directory with the given filename.
    :param img:
    :param directory: directory to save the image to.
    :param fname: image filename. Must include file type, eg. "image.png"
    :return:
    """

    io.imsave(os.path.join(directory, fname), img_as_ubyte(img))


def analyze_clusters(img, blobs):
    """
    Analyzes the clusters in a single image.
    :param img: single image with the clusters
    :param blobs: blobs obtained from img_watershed.
    :return: tuple - (image of clusters, pandas dataframe of cluster data)
    """

    clusters = np.zeros(img.shape, np.uint8)
    cl_area = []
    cl_perimeter = []
    cl_center = []
    cl_circularity = []
    defect_len_avg = []
    defect_len_std = []
    defect_len_min = []
    defect_len_max = []

    # loop over the unique labels returned by the img_watershed function
    for particle in np.unique(blobs):
        # if the label is zero, we are examining the 'background', so ignore it
        if particle == 0:
            continue
        # otherwise, allocate memory for the label region and draw it on the mask
        mask = np.zeros(img.shape, np.uint8)
        mask[blobs == particle] = 255
        # detect contours in the mask and grab the largest one
        try:
            cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            ct_im, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(clusters, cnts, 0, 255, -1)
        cl_area.append(cv2.contourArea(cnts[0]))
        cl_perimeter.append(cv2.arcLength(cnts[0], 1))
        M = cv2.moments(cnts[0], 0)
        if cl_area[-1] != 0:
            cl_center.append(tuple([int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])]))
        else:
            cl_center.append("None")

        # calculate circularity
        cl_circularity.append(4 * np.pi * (cv2.contourArea(cnts[0])) / (cv2.arcLength(cnts[0], 1) ** 2))

        # find the convex hull of the particle, and extract the defects
        cnt = cnts[0]
        hull = cv2.convexHull(cnt, returnPoints=False)
        dhull = cv2.convexHull(cnt, returnPoints=True)
        defects = cv2.convexityDefects(cnt, hull)

        pt_defects = []
        if defects is not None:
            for j in range(defects.shape[0]):
                s, e, f, d = defects[j, 0]
                # start = tuple(cnt[s][0])
                # end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])

                # store the length from the defects to the hull
                pt_defects.append(cv2.pointPolygonTest(dhull, far, True))

        # store the mean and stdev of the defect length
        # if there are no defects, just store 0
        if len(pt_defects) == 0:
            defect_len_avg.append(0)
            defect_len_std.append(0)
            defect_len_min.append(0)
            defect_len_max.append(0)
        else:
            defect_len_avg.append(np.mean(pt_defects))
            defect_len_std.append(np.std(pt_defects))
            defect_len_min.append(min(pt_defects))
            defect_len_max.append(max(pt_defects))

    cluster_data = {'Area': cl_area,
                    'Perimeter': cl_perimeter,
                    'Center': cl_center,
                    'Circularity': cl_circularity,
                    'Average Defect Length': defect_len_avg,
                    'Stdev of Devect Length': defect_len_std,
                    'Min Defect Length': defect_len_min,
                    'Max Defect Length': defect_len_max
                    }
    cluster_df = pd.DataFrame(cluster_data)

    return img_as_ubyte(clusters), cluster_df


def analyze_exp(stack, im_titles, cluster_dfs=None, save_dfs=False, save_ims=False, save_dir=None, imtype='bmp', min_distance=7):
    """
    :param stack: skimage image collection of numpy array of images to be analyzed.
    :param im_titles: list of names of the images; each must be unique.
    :param cluster_dfs: (optional) list of all cluster dataframes from analyze_clusters.
    :param save_dfs: (optional) save the individual cluster dataframes.
    :param save_ims: (optional) save the binary cluster images.
    :param save_dir: (optional) directory to save dataframes and clusters to. Required if save_df and/or save_ims = True
    :return: pandas dataframe of overall cluster statistics for the experiment.
    Note: images saved using this function will attempt to save them into a '/Clusters/' folder.
    """

    """Extract individual particle data while applying the Watershed to the stack"""
    # overall image data
    particle_ct = []
    avg_area = []
    std_area = []
    median_area = []
    mode_area = []
    range_area = []
    std_peri = []
    avg_peri = []
    median_peri = []
    mode_peri = []
    range_peri = []
    avg_def_len = []
    std_def_len = []

    # for every image in the stack
    for i in tqdm(range(len(stack))):
        # if user didn't pass a list of dataframes for the clusters
        if cluster_dfs == None:
            # get the info for each cluster
            blobs = img_watershed(stack[i], min_distance)
            # analyze the clusters
            clusters, cluster_df = analyze_clusters(stack[i], blobs)
            # if the user wants to save the dataframes and clusters
            if save_ims == True:
                try:
                    io.imsave(os.path.join(save_dir, "Clusters", "") + str(im_titles) + '.', imtype, img_as_ubyte(clusters))
                except (NameError, ValueError, FileNotFoundError):
                    print('Please provide valid directory for the images to be saved to.')
            if save_dfs == True:
                try:
                    cluster_df.to_csv(os.path.join(save_dir, str(im_titles) + '.csv'))
                except (NameError, ValueError, FileNotFoundError):
                    print('Please provide valid directory for the dataframes to be saved to.')
        # if the user did pass a list of dataframes
        else:
            try:
                # use the cluster dataframes passed from user
                cluster_df = cluster_dfs[i]
            except (NameError, ValueError, TypeError):
                return print('Please pass a list of cluster dataframes corresponding to each image in the experiment.'
                             '\nAlternatively, pass cluster_dfs=None to autogenerate them.'
                             '\n Note: this will not return or save the individual cluster dataframes.')

        # pull the cluster data out of the dataframe
        cl_area = cluster_df['Area']
        cl_perimeter = cluster_df['Perimeter']
        defect_len_avg = cluster_df['Average Defect Length']

        # add the overall cluster data to the data lists for the well dataframe
        particle_ct.append(len(cl_area))
        avg_area.append(np.mean(cl_area))
        std_area.append(np.std(cl_area))
        median_area.append(np.median(cl_area))
        mode_area.append(stats.mode(cl_area))
        range_area.append(np.ptp(cl_area))
        avg_peri.append(np.mean(cl_perimeter))
        std_peri.append(np.std(cl_perimeter))
        median_peri.append(np.median(cl_perimeter))
        mode_peri.append(stats.mode(cl_perimeter))
        range_peri.append(np.ptp(cl_perimeter))
        avg_def_len.append(np.mean(defect_len_avg))
        std_def_len.append(np.std(defect_len_avg))

    well_data = {'Image': im_titles,
                 'Particle Count': particle_ct,
                 'Average Area': avg_area,
                 'St Dev Area': std_area,
                 'Median Area': median_area,
                 'Mode Area': mode_area,
                 'Range Area': range_area,
                 'Average Perimeter': avg_peri,
                 'St Dev Perimeter': std_peri,
                 'Median Perimeter': median_peri,
                 'Mode Perimeter': mode_peri,
                 'Range Perimeter': range_peri,
                 'Average Def. Len.': avg_def_len,
                 'St Dev Def. Len': std_def_len
                 }
    well_df = pd.DataFrame(well_data)

    return well_df


def NASA_settling_time(im_name):
    """
    Accepts the filename of an image with the NASA naming convention for ACEH2, and returns the settling time.
    For use as image naming UDF.
    """

    s = im_name[-33:]
    start = s.find("E", s.find("\\"))
    end = s.find("_")
    t_settling = int(s[start + 1:end])
    return t_settling


def default_analysis(exp_dir, imtype='bmp', min_imgs=0, im_namer=None, block_size=71, offset=5, min_distance=7):
    """
    Runs full analysis on all images in nested folders.
    Saves images of clusters and corresponding csv's into each folder with images.
    :param exp_dir: full directory to main folder with experiment images.
    :param imtype: type of image; eg. bmp, jpg, png. Do NOT include '.' before the abbreviation.
    :param min_imgs:
    :param im_namer:
    :param block_size:
    :param offset: see loc_threshold
    :param min_distance: see loc_threshold
    :return:
    """

    """
    Runs full analysis on all images in nexted folders
    Saves images of clusters and corresponding csv's into each folder with images.
    Arguments:
        exp_dir - full directory to main folder with experimental images.
        imtype - type of image; eg. bmp, jpg, png.
        min_imgs - minimum number of images in a folder to be analyzed. Default is 0.
        im_namer - UDF to assign titles to an image. Default will name images 0 to length of stack in order of analysis.
    """

    for entry in exp_dir.iterdir():
        if entry.is_dir():
            imgs = io.ImageCollection(os.path.join(entry, '*.' + str(imtype)))

            if len(imgs) >= min_imgs and entry.name != 'Clusters' and entry.name != 'ROIs':
                data = {'Filename': imgs.files}
                files_df = pd.DataFrame(data)
                files_df.to_csv(str(entry) + ' Images.csv')
                print(str(entry))

                exp = entry.name
                # Create the folder to hold copies of all of the ROIs
                Path(str(entry) + "/ROIs").mkdir(parents=True, exist_ok=True)
                # Create folders to hold the clusters obtained from each ROI
                Path(str(entry) + "/Clusters").mkdir(parents=True, exist_ok=True)

                # get the settling times (in sec) for each image
                if im_namer == None:
                    im_titles = [n for n in len(imgs.files)]
                else:
                    try:
                        im_titles = [im_namer(i) for i in imgs.files]
                    except ValueError:
                        print('Please provide a valid UDF, or pass im_namer=None')
                        exit

                # set up the dataframe to hold the overall well data
                exp_dict = {'Filename': imgs.files,
                            'Image name': im_titles
                            }
                exp_df = pd.DataFrame(exp_dict)

                """
                Execute Image Processing
                """

                # crop images, and save them
                rois = crop_imgs(imgs)
                for i in range(len(rois)):
                    saveimg(img_as_ubyte(rois[i]), entry, os.path.join("ROIs", "") + str(im_titles[i]) + '.' + str(imtype))

                # Convert to rois to binary
                binary_rois = loc_threshold(rois, block_size, offset)

                # Remove single particles and gritty imperfections
                clean_rois = clean_imgs(binary_rois)

                # Analyze the well
                overall_df = analyze_exp(clean_rois, im_titles, save_dfs=True, save_ims=True, save_dir=entry, min_distance=7)

                exp_df = pd.merge(exp_df, overall_df, on='Image name')
                exp_df.to_csv(os.path.join(entry, exp + ' Data.csv'))

            run_analysis(entry)
