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
            while not keyboardClick:
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
            self.cropped = CspyStack(
                [self[i][tpleft[0]:btmright[0], tpleft[1]:btmright[1]] for i in range(len(self))])
        elif len(self.shape) == 2:
            self.cropped = CspyStack(self[tpleft[0]:btmright[0], tpleft[1]:btmright[1]])
        else:
            raise Exception

    def add_otsu(self, nbins=256):
        if self.cropped is None:
            if len(self.shape) == 3:
                self.binary_otsu = []
                for i in tqdm(range(len(self)), desc='Applying otsu threshold to cspy_stack.binary_otsu', leave=True):
                    otsu = filters.threshold_otsu(self[i], nbins=nbins)
                    self.binary_otsu.append(img_as_ubyte(self[i] > otsu))
            elif len(self.shape) == 2:
                print('Applying otsu threshold to cspy_stack.binary_otsu')
                otsu = filters.threshold_otsu(self, nbins=nbins)
                self.binary_otsu = img_as_ubyte(self > otsu)
            else:
                raise Exception
        else:
            if len(self.shape) == 3:
                self.binary_otsu = []
                for i in tqdm(range(len(self)), desc='Applying otsu threshold to cspy_stack.binary_otsu', leave=True):
                    otsu = filters.threshold_otsu(self.cropped[i], nbins=nbins)
                    self.binary_otsu.append(img_as_ubyte(self.cropped[i] > otsu))
            elif len(self.shape) == 2:
                print('Applying otsu threshold to cspy_stack.binary_otsu')
                otsu = filters.threshold_otsu(self.cropped, nbins=nbins)
                self.binary_otsu = img_as_ubyte(self.cropped > otsu)
            else:
                raise Exception

    def add_local_threshold(self, block_size=71, offset=5, cutoff=0, **kwargs):
        if self.cropped is None:
            if len(self.shape) == 3:
                self.binary_loc = []
                for i in tqdm(range(len(self)), desc='Applying local threshold to cspy_stack.binary_loc', leave=True):
                    local_thresh = filters.threshold_local(self[i], block_size=block_size, offset=offset, **kwargs)
                    low_val_flags = local_thresh < cutoff
                    local_thresh[low_val_flags] = 255
                    self.binary_loc.append(img_as_ubyte(self[i] > local_thresh))
            elif len(self.shape) == 2:
                print('Applying local threshold to cspy_stack.binary_loc')
                local_thresh = filters.threshold_local(self, block_size=block_size, offset=offset, **kwargs)
                low_val_flags = local_thresh < cutoff
                local_thresh[low_val_flags] = 255
                self.binary_loc = img_as_ubyte(self > local_thresh)
            else:
                raise Exception
        else:
            if len(self.shape) == 3:
                self.binary_loc = []
                for i in tqdm(range(len(self)), desc='Applying local threshold to cspy_stack.binary_loc', leave=True):
                    local_thresh = filters.threshold_local(self.cropped[i], block_size=block_size, offset=offset, **kwargs)
                    low_val_flags = local_thresh < cutoff
                    local_thresh[low_val_flags] = 255
                    self.binary_loc.append(img_as_ubyte(self.cropped[i] > local_thresh))
            elif len(self.shape) == 2:
                print('Applying local threshold to cspy_stack.binary_loc')
                local_thresh = filters.threshold_local(self.cropped, block_size=block_size, offset=offset, **kwargs)
                low_val_flags = local_thresh < cutoff
                local_thresh[low_val_flags] = 255
                self.binary_loc = img_as_ubyte(self.cropped > local_thresh)
            else:
                raise Exception
        # self.binary_loc = self.binary_loc.astype(np.uint8)

    def add_hysteresis_threshold(self, low=20, high=150):
        if self.cropped is None:
            if len(self.shape) == 3:
                self.binary_hyst = []
                for i in tqdm(range(len(self)),
                              desc='Applying hysteresis threshold to cspy_stack.binary_hyst', leave=True):
                    self.binary_hyst.append(filters.apply_hysteresis_threshold(self[i], low=low, high=high))
            elif len(self.shape) == 2:
                print('Applying hysteresis threshold to cspy_stack.binary_hyst')
                self.binary_hyst = filters.apply_hysteresis_threshold(self, low=low, high=high)
            else:
                raise Exception
        else:
            if len(self.shape) == 3:
                self.binary_hyst = []
                for i in tqdm(range(len(self)),
                              desc='Applying hysteresis threshold to cspy_stack.binary_hyst', leave=True):
                    self.binary_hyst.append(filters.apply_hysteresis_threshold(self.cropped[i], low=low, high=high))
            elif len(self.shape) == 2:
                print('Applying hysteresis threshold to cspy_stack.binary_hyst')
                self.binary_hyst = filters.apply_hysteresis_threshold(self.cropped, low=low, high=high)
            else:
                raise Exception

    def add_cleaned(self, bin_stack):
        if len(self.shape) == 3:
            self.cleaned = []
            for i in tqdm(range(len(self)), desc='Adding cleaned to cspy_stack.cleaned', leave=True):
                self.cleaned.append(img_as_ubyte(ndimage.binary_closing(ndimage.binary_opening(bin_stack[i]))))
        elif len(self.shape) == 2:
            print('Adding cleaned to cspy_stack.cleaned')
            self.cleaned = img_as_ubyte(ndimage.binary_closing(ndimage.binary_opening(bin_stack)))
        else:
            raise Exception

    def find_particles(self, bin_stack, min_distance=7):
        if type(bin_stack) == list or len(bin_stack.shape) == 3:
            self.particles = []
            for i in tqdm(range(len(bin_stack)), desc='Adding detected particles to cspy_stack.particles', leave=True):
                distance = ndimage.distance_transform_edt(bin_stack[i])
                local_max = peak_local_max(distance, min_distance=min_distance, labels=bin_stack[i])
                local_max_mask = np.zeros(distance.shape, dtype=bool)
                local_max_mask[tuple(local_max.T)] = True
                markers = ndimage.label(local_max_mask)[0]
                labels = CspyStack(watershed(-distance, markers, mask=bin_stack[i]))
                self.particles.append(labels)
        elif len(bin_stack.shape) == 2:
            print('Adding detected particles to cspy_stack.particles')
            distance = ndimage.distance_transform_edt(bin_stack)
            local_max = peak_local_max(distance, min_distance=min_distance, labels=bin_stack)
            local_max_mask = np.zeros(distance.shape, dtype=bool)
            local_max_mask[tuple(local_max.T)] = True
            markers = ndimage.label(local_max_mask)[0]
            labels = watershed(-distance, markers, mask=bin_stack)
            self.particles = CspyStack(labels)
        else:
            raise Exception

    def analyze_particles(self, particles, im=None, min_area=0):
        def single(image, particles, min_area):
            clusters = np.zeros(image.shape, np.uint8)
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
                mask = np.zeros(image.shape, np.uint8)
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
                dhull = cv2.convexHull(cnt, returnPoints=True)
                defects = cv2.convexityDefects(cnt, hull)

                pt_defects = []
                if defects is not None:
                    for j in range(defects.shape[0]):
                        s, e, f, d = defects[j, 0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])
                        tfar = tuple(map(int, far))

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

        if im is None:
            # single returns BOTH the image of clusters and dataframe as a tuple
            if len(self.shape) == 3:
                self.particle_data = [single(self.cleaned[i], particles[i], min_area=min_area)[1]
                                      for i in tqdm(range(len(self)),
                                                    desc='Populating cspy_stack.particle_data', leave=True)]
            elif len(self.shape) == 2:
                print('Populating cspy_stack.particle_data')
                self.particle_data = single(self.cleaned, particles, min_area)[1]
            else:
                raise Exception
        else:
            if len(im) == 3:
                self.particle_data = [single(self.cleaned[i], particles[i], min_area=min_area)[1]
                                      for i in tqdm(range(len(self)),
                                                    desc='Populating cspy_stack.particle_data', leave=True)]
            elif len(self.shape) == 2:
                print('Populating cspy_stack.particle_data')
                self.particle_data = single(self.cleaned, particles, min_area)
            else:
                raise Exception

    def analyze_stack(self, im_titles=None, save_dfs=False, save_ims=False, save_dir=None, imtype='tiff',
                      min_distance=7):
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

        if im_titles is None:
            im_titles = np.arange(len(self))
        if self.cleaned is None:
            raise Exception('Generate cleaned images first using cspy_stack.find_particles(binary_stack)'
                            '\n attributes .binary_otsu, .binary_loc, or .binary_hyst may be used')
        if self.particles is None:
            self.find_particles(self.cleaned, min_distance)

        for i in tqdm(range(len(self)), desc='Analyzing particles', leave=True):
            clusters, cluster_df = self.analyze_particles(self.cleaned[i], self.particles[i])
            # if user wants to save the dataframes and clusters
            if save_ims:
                try:
                    Path(str(save_dir) + '/Clusters').mkdir(parents=True, exist_ok=True)
                    io.imsave(os.path.join(save_dir, "Clusters", im_titles[i] + '.' + imtype),
                              img_as_ubyte(clusters))
                except (NameError, ValueError, FileNotFoundError):
                    print('Please provide valid directory for the images to be saved to.')
            if save_dfs:
                try:
                    cluster_df.to_csv(os.path.join(save_dir, str(im_titles[i]) + '.csv'))
                except (NameError, ValueError, FileNotFoundError):
                    print('Please provide valid directory for particle dataframes to be saved to.')

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
                      'Average Def. Len.': avg_def_len,
                      'St Dev Def. Len': std_def_len
                      }
        stack_df = pd.DataFrame(stack_data)

        return stack_df


if __name__ == '__main__':
    os.chdir('/home/adam/Documents/ACET12 2020/Temp Control/C1/X20/Week 3/SR5/Y19/')
    # os.chdir('C:/Users/Adam/Desktop/NASA Data/ACET12/DR5_20C/Y107/')
    stack = CspyStack(*load('20210406_132925.122_CnFcl_ACET12_S2020_C1_X20_SR5_Y19_00008.tiff'))
    # stack.add_cropped()
    # stack.add_local_threshold(block_size=151, offset=1, cutoff=3)
    stack.add_otsu()
    stack.add_cleaned(stack.binary_otsu)
    stack.find_particles(stack.cleaned, min_distance=3)
    io.imshow(stack.particles[0], cmap='gray')
    # io.imshow(stack.view_particles(stack[0], stack.particles[0], weight=1))
    # from colloidspy.analyze import analyze_clusters
    # import seaborn as sns
    stack.analyze_particles(stack.particles)
    df = stack.particle_data[0]
    # df['Area (um^2)'] = df['Area'] * (0.3**2)
    # sns.displot(df, x='Area (um^2)', bins=100).set(xlim=[0, 25])
    # plt.tight_layout()





"""
Installation bug fixes on Ubuntu:
- opencv must be opencv-python-headless unless you go through the whole build process
- skimage/mpl must have PyQt(5) or a similar thing installed to get matplotlib in GUI mode
- skimage/mpl must have python-kt (TkAgg) installed for certain plotting functions
"""