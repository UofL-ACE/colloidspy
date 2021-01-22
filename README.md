# Colloidspy

This python package is designed to assist in analyzing confocal images of colloidal systems.
The motivation for the package arose when attempting to analyze hundreds of images for NASA's Advanced Colloids Experiments (ACE).
The lack of robust, flexible, automated methods to analyze large stacks of images of highly dense colloidal systems blocked the ability to extract meaning and accurate results from one of the experiments conducted on the ISS in 2016 without creating new a new analysis tool.
Colloidspy incorporates robust filters, edge detection, and contour description from scikit-image and opencv python packages, and bundles them into convenient functions that can be used to extract 2D cluster data from colloidal settling images with a minimal learning curve.
It is designed to be used alongside pandas and numpy to control structuring and export of data.

## Example
```python
import colloidspy as cspy
import skimage.io as io
import numpy as np
import os

directory = 'replace with the directory that the images are in'
stack = io.imread_collection(os.path.join(directory, '*.bmp'))
cropped = cspy.crop_imgs(stack)
# in high density systems, particularly with lighting gradients, a local threshold will give the best results.
binary_stack = cspy.loc_threshold(cropped, block_size=71, offset=5, cutoff=0, method='gaussian')
# in less dense systems and systems with even lighting, and global threshold (otsu) may be give better results.
binary_stack_alt = cspy.otsu_threshold(cropped)
# BEFORE RUNNING ANALYSIS, test a handful of your images with loc_threshold and otsu_threshold to see which works
# better for your system, and to determine the optimal parameters (block_size, offset, etc)

# remove gritty imperfections from the images
cleaned_stack = cspy.clean_imgs(binary_stack)

# to only analyze a single image:
blobs = cspy.img_watershed(clean_stack[im])
clusters, cl_data = cspy.analyze_clusters(clean_stack[im], blobs)

# to analyze all images in the stack, and save all images of the clusters and csv's of all of the cluster dataframes:
# if you do not want to save the analyzed images and their dataframes, set save_dfs=False, save_ims=False.
# alternatively, you can pass a list of pandas dataframes of the cluster data if you have already generated them.
im_titles = ['list of image titles to use to save the analyzed images and datafiles']
overall_df = analyze_exp(cleaned_stack, im_titles, cluster_dfs=False, save_dfs=True, save_ims=True, save_dir=directory)

# finally, to run the version of the analysis that worked for the system this package was designed for:
run_analysis(exp_dir, imtype='bmp', min_imgs=0, im_namer=None, block_size=71, offset=5, min_distance=7)
```

