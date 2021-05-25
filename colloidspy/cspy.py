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
# %matplotlib qt
from tqdm import tqdm


def load(path, conserve_memory=True):
    stack = io.ImageCollection(path, conserve_memory=conserve_memory)
    if len(np.shape(stack)) == 4:
        from skimage.color import rgb2gray
        if np.shape(stack)[-1] == 4:
            from skimage.color import rgba2rgb
            files = stack.files
            stack = [rgb2gray(rgba2rgb(stack[i])) for i in range(len(stack))]
        else:
            files = stack.files
            stack = [rgb2gray(stack[i]) for i in range(len(stack))]
        return img_as_ubyte(stack), files
    else:
        return img_as_ubyte(stack), stack.files


def view_particles(img, particles, min_area=0, fill=False, weight=0):
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
            cv2.drawContours(clusters, cnts, 0, (255, 0, 0), weight)
        else:
            cv2.drawContours(clusters, cnts, 0, (255, 0, 0), weight)
    return clusters


class CspyStack(np.ndarray):
    """
    Class to hold raw and process images and their respective particle data.
    Attributes:
        cropped - image ROI's, set with add_cropped()
        binary_otsu - ROI's binarized with otsu threshold, set with add_otsu()
        binary_loc - ROI's binarized with local threshold, set with add_local_threshold()
        binary_hyst - ROI's binarized with hysteresis threshold, set with add_hysteresis_threshold()
        cleaned - Cleaned binary ROI's, set with add_cleaned(BinaryStack) where BinaryStack is the set of binary
                    images chosen earlier with otsu, local, or hysteresis threshold methods
        particles - ROI where the pixels of all particles are assigned a unique integer labeled by particle
        particle_data - pandas dataframes of particle data in each image
        crop_coords - (x,y) coordinates of image ROI
    """
    def __new__(cls, input_array, filenames=None, cropped=None, crop_coords=None, binary_otsu=None, binary_loc=None,
                binary_hyst=None, cleaned=None, particles=None, particle_data=None):
        obj = np.asarray(input_array).view(cls)
        obj.filenames = filenames
        obj.cropped = cropped
        obj.crop_coords = crop_coords
        obj.binary_otsu = binary_otsu
        obj.binary_loc = binary_loc
        obj.binary_hyst = binary_hyst
        obj.cleaned = cleaned
        obj.particles = particles
        obj.particle_data = particle_data
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
        self.particle_data = getattr(obj, 'particle_data', None)

    def add_cropped(self, cropall=True):
        """
        Interactictive image cropper, built from matplotlib.
        Select ROI from top-left to bottom-right with mouse, press any key to accept selection except q and r
        Sets CspyStack.cropped attribute
        :param cropall: True - Same ROI is applied to all images in stack. False - crop all images individually
        :return: nothing
        """
        def line_select_callback(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            while erelease is None:
                pass
            print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
            print(" The button you used were: %s %s" % (eclick.button, erelease.button))
            self.crop_coords.append(((int(x1), int(y1)), (int(x2), int(y2))))

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
            # # some IDE's (pycharm, jupyter) move on to plt.close(fig) instead of waiting for the callback to finish
            # # if that is the case, uncomment the next three lines
            keyboardClick = False
            while not keyboardClick:
                keyboardClick = plt.waitforbuttonpress()
            plt.close(fig)

        self.crop_coords = []
        if len(self.shape) == 2:
            img = self
            select_roi(img)
            # Crop all images in the stack
            # Numpy's axis convention is (y,x) or (row,col)
            tpleft = [self.crop_coords[0][0][1], self.crop_coords[0][0][0]]
            btmright = [self.crop_coords[0][1][1], self.crop_coords[0][1][0]]
            self.cropped = CspyStack(self[tpleft[0]:btmright[0], tpleft[1]:btmright[1]])

        elif len(self.shape) == 3:
            if cropall:
                img = self[0]
                select_roi(img)
                tpleft = [self.crop_coords[0][0][1], self.crop_coords[0][0][0]]
                btmright = [self.crop_coords[0][1][1], self.crop_coords[0][1][0]]
                self.cropped = CspyStack([self[i][tpleft[0]:btmright[0], tpleft[1]:btmright[1]]
                                          for i in range(len(self))])
            else:
                cropped_imgs = []
                for i in range(len(self)):
                    img = self[i]
                    select_roi(img)
                    tpleft = [self.crop_coords[i][0][1], self.crop_coords[i][0][0]]
                    btmright = [self.crop_coords[i][1][1], self.crop_coords[i][1][0]]
                    cropped_imgs.append(self[i][tpleft[0]:btmright[0], tpleft[1]:btmright[1]])
                self.cropped = CspyStack(cropped_imgs)
        else:
            raise Exception("TypeError in add_cropped - is the stack greyscale?")

    def add_otsu(self, nbins=256):
        """
        Adds attribute binary_otsu from CspyStack.cropped if available. Otherwise, uses raw stack.
        :param nbins: number of bins in image histogram
        :return: nothing
        """
        if self.cropped is None:
            if len(self.shape) == 3:
                binary = []
                for i in tqdm(range(len(self)), desc='Applying otsu threshold to CspyStack.binary_otsu', leave=True):
                    otsu = filters.threshold_otsu(self[i], nbins=nbins)
                    binary.append(img_as_ubyte(self[i] > otsu))
                self.binary_otsu = CspyStack(binary)
            elif len(self.shape) == 2:
                print('Applying otsu threshold to CspyStack.binary_otsu')
                otsu = filters.threshold_otsu(self, nbins=nbins)
                self.binary_otsu = CspyStack(img_as_ubyte(self > otsu))
            else:
                raise Exception
        else:
            if len(self.shape) == 3:
                binary = []
                for i in tqdm(range(len(self)), desc='Applying otsu threshold to CspyStack.binary_otsu', leave=True):
                    otsu = filters.threshold_otsu(self.cropped[i], nbins=nbins)
                    binary.append(img_as_ubyte(self.cropped[i] > otsu))
                self.binary_otsu = CspyStack(binary)
            elif len(self.shape) == 2:
                print('Applying otsu threshold to CspyStack.binary_otsu')
                otsu = filters.threshold_otsu(self.cropped, nbins=nbins)
                self.binary_otsu = CspyStack(img_as_ubyte(self.cropped > otsu))
            else:
                raise Exception('TypeError: shape of images not correct. Is stack greyscale?')

    def add_local_threshold(self, block_size=71, offset=5, cutoff=0, **kwargs):
        """
        Adds attribute binary_loc from CspyStack.cropped if available. Otherwise, uses raw stack.
        See https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_local
        :param block_size: Odd size of pixel neighborhood which is used to calculate the threshold value
        :param offset: Constant subtracted from weighted mean of neighborhood to calculate the local threshold value.
        :param cutoff: lowest pixel value in gaussian image to be considered in threshold. Useful if large black areas
                        are showing up as white areas in the thresholded image
        :param kwargs: other kwargs from skimage threshold_local() function
        :return:
        """
        if self.cropped is None:
            if len(self.shape) == 3:
                binary = []
                for i in tqdm(range(len(self)), desc='Applying local threshold to CspyStack.binary_loc', leave=True):
                    local_thresh = filters.threshold_local(self[i], block_size=block_size, offset=offset, **kwargs)
                    low_val_flags = local_thresh < cutoff
                    local_thresh[low_val_flags] = 255
                    binary.append(img_as_ubyte(self[i] > local_thresh))
                self.binary_loc = CspyStack(binary)
            elif len(self.shape) == 2:
                print('Applying local threshold to CspyStack.binary_loc')
                local_thresh = filters.threshold_local(self, block_size=block_size, offset=offset, **kwargs)
                low_val_flags = local_thresh < cutoff
                local_thresh[low_val_flags] = 255
                self.binary_loc = CspyStack(img_as_ubyte(self > local_thresh))
            else:
                raise Exception('TypeError: shape of images not correct. Is stack greyscale?')
        else:
            if len(self.shape) == 3:
                binary = []
                for i in tqdm(range(len(self)), desc='Applying local threshold to CspyStack.binary_loc', leave=True):
                    local_thresh = filters.threshold_local(self.cropped[i], block_size=block_size, offset=offset, **kwargs)
                    low_val_flags = local_thresh < cutoff
                    local_thresh[low_val_flags] = 255
                    binary.append(img_as_ubyte(self.cropped[i] > local_thresh))
                self.binary_loc = CspyStack(binary)
            elif len(self.shape) == 2:
                print('Applying local threshold to CspyStack.binary_loc')
                local_thresh = filters.threshold_local(self.cropped, block_size=block_size, offset=offset, **kwargs)
                low_val_flags = local_thresh < cutoff
                local_thresh[low_val_flags] = 255
                self.binary_loc = CspyStack(img_as_ubyte(self.cropped > local_thresh))
            else:
                raise Exception('TypeError: shape of images not correct. Is stack greyscale?')

    def add_hysteresis_threshold(self, low=20, high=150):
        """
        Adds attribute binary_hyst from CspyStack.cropped if available, otherwise uses raw stack.
        See https://scikit-image.org/docs/dev/auto_examples/filters/plot_hysteresis.html
        Pixels above the high theshold are considered to be a particle, pixels between low and high values are only
        considered part of a particle if they touch another particle that was designated as a particle.
        :param low: lowest pixel value to be considered as part of a potential particle
        :param high: pixel values higher than this threshold area always considered a particle
        :return: nothing
        """
        if self.cropped is None:
            if len(self.shape) == 3:
                binary = []
                for i in tqdm(range(len(self)),
                              desc='Applying hysteresis threshold to CspyStack.binary_hyst', leave=True):
                    binary.append(filters.apply_hysteresis_threshold(self[i], low=low, high=high))
                self.binary_hyst = CspyStack(binary)
            elif len(self.shape) == 2:
                print('Applying hysteresis threshold to CspyStack.binary_hyst')
                self.binary_hyst = CspyStack(filters.apply_hysteresis_threshold(self, low=low, high=high))
            else:
                raise Exception('TypeError: shape of images not correct. Is stack greyscale?')
        else:
            if len(self.shape) == 3:
                binary = []
                for i in tqdm(range(len(self)),
                              desc='Applying hysteresis threshold to CspyStack.binary_hyst', leave=True):
                    binary.append(filters.apply_hysteresis_threshold(self.cropped[i], low=low, high=high))
                self.binary_hyst = CspyStack(binary)
            elif len(self.shape) == 2:
                print('Applying hysteresis threshold to CspyStack.binary_hyst')
                self.binary_hyst = CspyStack(filters.apply_hysteresis_threshold(self.cropped, low=low, high=high))
            else:
                raise Exception('TypeError: shape of images not correct. Is stack greyscale?')

    def add_cleaned(self, bin_stack):
        """
        Adds attribute CspyStack.cleaned
        Removes small roughly single-pixel imperfections leaving only the major structuring elements.
        Uses a binary opening and closing algorithm
        :param bin_stack: binary images to clean. Use one of the three binary image attributes (otsu, loc, hyst)
        :return:
        """
        if len(self.shape) == 3:
            self.cleaned = []
            for i in tqdm(range(len(self)), desc='Adding cleaned to CspyStack.cleaned', leave=True):
                self.cleaned.append(img_as_ubyte(ndimage.binary_closing(ndimage.binary_opening(bin_stack[i]))))
        elif len(self.shape) == 2:
            print('Adding cleaned to CspyStack.cleaned')
            self.cleaned = img_as_ubyte(ndimage.binary_closing(ndimage.binary_opening(bin_stack)))
        else:
            raise Exception

    def find_particles(self, bin_stack, min_distance=7):
        """
        Adds attribute CspyStack.particles
        Applies a watershed algorithm to the binary stack (preferably cleaned) to identify distinct particles.
        Pixels are assigned an integer, unique by particle, where 0 is the background
        :param bin_stack: stack of binary images with particles to detect
        :param min_distance: minimum distance in pixels between particle centers
        :return: nothing
        """
        if type(bin_stack) == list or len(bin_stack.shape) == 3:
            particles = []
            for i in tqdm(range(len(bin_stack)), desc='Adding detected particles to CspyStack.particles', leave=True):
                distance = ndimage.distance_transform_edt(bin_stack[i])
                local_max = peak_local_max(distance, min_distance=min_distance, labels=bin_stack[i])
                local_max_mask = np.zeros(distance.shape, dtype=bool)
                local_max_mask[tuple(local_max.T)] = True
                markers = ndimage.label(local_max_mask)[0]
                labels = watershed(-distance, markers, mask=bin_stack[i])
                particles.append(labels)
            self.particles = CspyStack(particles)
        elif len(bin_stack.shape) == 2:
            print('Adding detected particles to CspyStack.particles')
            distance = ndimage.distance_transform_edt(bin_stack)
            local_max = peak_local_max(distance, min_distance=min_distance, labels=bin_stack)
            local_max_mask = np.zeros(distance.shape, dtype=bool)
            local_max_mask[tuple(local_max.T)] = True
            markers = ndimage.label(local_max_mask)[0]
            labels = watershed(-distance, markers, mask=bin_stack)
            self.particles = CspyStack(labels)
        else:
            raise Exception

    def analyze_particles(self, particles, min_area=0):
        """
        Adds attribute CspyStack.particle_data
        :param particles: stack of watershed images where particles are labeled by a unique integer
        :param min_area: minimum particle area to be considered a particle
        :return: list of pandas dataframes of particle data, or single dataframe if only one image was passed
        """
        def single(particles, min_area):
            # clusters = np.zeros(image.shape, np.uint8)
            clusters = np.zeros(particles.shape, np.uint8)
            cl_area = []
            cl_perimeter = []
            cl_center = []
            cl_circularity = []
            defect_len_avg = []
            defect_len_std = []
            defect_len_min = []
            defect_len_max = []

            for particle in np.unique(particles):
                if particle == 0:
                    continue
                # mask = np.zeros(image.shape, np.uint8)
                mask = np.zeros(particles.shape, np.uint8)
                mask[particles == particle] = 255
                try:
                    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                except ValueError:
                    # older version of opencv returns three objects instead of two
                    ct_im, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cv2.contourArea(cnts[0]) < min_area:
                    continue
                cv2.drawContours(clusters, cnts, 0, 255, -1)
                cl_area.append(cv2.contourArea(cnts[0]))
                cl_perimeter.append(cv2.arcLength(cnts[0], 1))
                M = cv2.moments(cnts[0], 0)
                if cl_area[-1] != 0:
                    cl_center.append(tuple([int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])]))
                    cl_circularity.append(4 * np.pi * (cv2.contourArea(cnts[0])) / (cv2.arcLength(cnts[0], 1) ** 2))
                else:
                    cl_center.append("None")
                    cl_circularity.append("None")

                # find the convex hull of the particle, and extract the defects
                cnt = cnts[0]
                hull = cv2.convexHull(cnt, returnPoints=False)
                # dhull = cv2.convexHull(cnt, returnPoints=True)
                defects = cv2.convexityDefects(cnt, hull)

                pt_defects = []
                if defects is not None:
                    for j in range(defects.shape[0]):
                        s, e, f, d = defects[j, 0]
                        # start = tuple(cnt[s][0])
                        # end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])
                        tfar = tuple(map(int, far))  # current(?) v of opencv-python doesnt play well with np.int32

                        # store the length from the defects to the hull
                        pt_defects.append(cv2.pointPolygonTest(cnt, tfar, True))

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
                            'Stdev of Defect Length': defect_len_std,
                            'Min Defect Length': defect_len_min,
                            'Max Defect Length': defect_len_max
                            }
            cluster_df = pd.DataFrame(cluster_data)
            return img_as_ubyte(clusters), cluster_df

        # single returns BOTH the image of clusters and dataframe as a tuple
        if len(self.shape) == 3:
            self.particle_data = [single(particles[i], min_area=min_area)[1] for i in tqdm(
                range(len(self)), desc='Populating CspyStack.particle_data', leave=True)]
        elif len(self.shape) == 2:
            print('Populating CspyStack.particle_data')
            self.particle_data = single(particles, min_area)[1]
        else:
            raise Exception

        return self.particle_data

    def analyze_stack(self, save_dfs=False, save_ims=False, save_dir=None, im_titles=None, imtype='tiff',
                      min_distance=7):
        """
        analyzes full stack of images, sets particles and particle_data attributes
        :param im_titles: optional, list of titles for each image and df to be saved with. Default is numbered from 0
        :param save_dfs: Boolean, set to true to save dataframes in the directory provided in save_dir
        :param save_ims: Boolean, set to true to save binary imgs of the particles in the directory provided in save_dir
        :param save_dir: directory (string) to save df's and/or images to. Required if saving df's or images
        :param imtype: type of image to save binary images as. Default is tiff
        :param min_distance: minimum distance in pixels between particle centers
        :return: single dataframe of overall particle data for each image, not saved to an attribute
        """
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

        print('Analyzing particles...')

        if im_titles is None:
            im_titles = np.arange(len(self))
        if self.cleaned is None:
            raise Exception('Generate cleaned images first using cspy_stack.find_particles(binary_stack)'
                            '\n attributes .binary_otsu, .binary_loc, or .binary_hyst may be used')
        if self.particles is None:
            self.find_particles(self.cleaned, min_distance)

        if self.particle_data is None:
            self.analyze_particles(self.particles)

        for i in tqdm(range(len(self)), desc='Saving particles and creating summary', leave=True):
            cluster_df = self.particle_data[i]
            # if user wants to save the dataframes and clusters
            if save_ims:
                try:
                    if save_dir is None:
                        Path('Clusters').mkdir(parents=True, exist_ok=True)
                        io.imsave(os.path.join("Clusters", str(im_titles[i]) + '.' + imtype),
                                img_as_ubyte(self.cleaned[i]))                    
                    else:
                            Path(str(save_dir) + '/Clusters').mkdir(parents=True, exist_ok=True)
                            io.imsave(os.path.join(save_dir, "Clusters", str(im_titles[i]) + '.' + imtype),
                                        img_as_ubyte(self.cleaned[i]))
                except (NameError, ValueError, FileNotFoundError):
                    print('Please provide valid directory for the images to be saved to, using kwarg save_dir')
            if save_dfs:
                try:
                    if save_dir is None:
                        Path('Data').mkdir(parents=True, exist_ok=True)
                        cluster_df.to_csv(os.path.join('Data', str(im_titles[i]) + '.csv'))
                    else:
                        cluster_df.to_csv(os.path.join(save_dir, 'Data', str(im_titles[i]) + '.csv'))
                except (NameError, ValueError, FileNotFoundError):
                    print('Please provide valid directory for particle dataframes to be saved to, using kwarg save_dir')

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

        stack_data = {'Image': im_titles,
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
                      'Average Defect Len.': avg_def_len,
                      'St Dev Defect Len': std_def_len
                      }
        stack_df = pd.DataFrame(stack_data)

        return stack_df

    def rdf(self, ind, res=300):
        """
        Calculates the 2D radial distribution function from the image at the index provided
        Returns a tuple: (distance/avg particle radius, rdf)
        """
        def distmatrix(centers):
            # makes an nxn matrix of distances between every cluster.
            # Diagonals are 0, upper and lower triangles are mirror images because distance from particle 2-1 is same as 1-2
            from scipy.spatial.distance import cdist
            return cdist(centers, centers, metric='euclidean')
        
        image = self.cleaned[ind]
        data = self.particle_data[ind]
        height, width = np.shape(image)
        n = len(data)
        distmat = distmatrix(data.Center.tolist())
        sigma = np.sqrt((np.mean(data.Area)/np.pi))

        # g(r) = <n>/dA = local density / bulk density
        d = np.sqrt(width**2 + height**2)
        bulk_density = n / (height*width)
        dn, edges = np.histogram(distmat[np.triu_indices(n, 1)], bins=res)
        da = np.pi * (edges[1:]**2 - edges[:-1]**2)
        loc_density = dn/da
        gr = loc_density/bulk_density
        plt.figure()
        plt.plot(edges[1:]/sigma, gr)
        plt.ylabel('g(r)')
        plt.xlabel('Distances/average particle radius')
        plt.xlim(0, d/sigma)
        plt.show()
        
        return edges[1:]/sigma, gr

    def structure_factor(self, ind):
        """
        Calculates the squared intensity of the DFT of an image.
        :param ind: index to the desired image
        :return: list: DFT intensity squared over spatial frequency in 1/px
        """
        import scipy.misc as spm
        from scipy.signal import medfilt
        from scipy import signal

        img = self[ind]

        def radial_profile(data, center):
            """
            Returns the sum of pixel values as a function of radius from the center of the given DFT.
            Created by Ben King, UofL, 2016
            """
            y, x = np.indices(data.shape)  # first determine radii of all pixels
            r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            ind = np.argsort(r.flat)  # get sorted indices
            sr = r.flat[ind]  # sorted radii
            sim = data.flat[ind]  # image values sorted by radii
            ri = sr.astype(np.int32)  # integer part of radii (bin size = 1)
            # determining distance between changes
            deltar = ri[1:] - ri[:-1]  # assume all radii represented
            rind = np.where(deltar)[0]  # location of changed radius
            nr = rind[1:] - rind[:-1]  # number in radius bin
            csim = np.cumsum(sim, dtype=np.float64)  # cumulative sum to figure out sums for each radii bin
            tbin = csim[rind[1:]] - csim[rind[:-1]]  # sum for image values in radius bins
            radialprofile = tbin / nr  # the answer
            return radialprofile

        H, W = np.shape(img)
        image_dft = np.fft.fft2(img)
        radpro = radial_profile(np.log(np.abs(np.fft.fftshift(image_dft))), (H/2, W/2))
        sos = signal.ellip(1, 0.009, 80, 0.015, output='sos')
        # sfactor = np.exp(signal.sosfilt(sos, radpro))
        sfactor = signal.sosfilt(sos, radpro)

        return sfactor
