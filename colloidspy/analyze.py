import os
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from skimage.util import img_as_ubyte
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage, stats
from tqdm import tqdm


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
