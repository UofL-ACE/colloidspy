# Colloidspy

This python package is designed to assist in analyzing confocal images of colloidal systems.
The motivation for the package arose when attempting to analyze hundreds of images for NASA's Advanced Colloids Experiments (ACE).
The lack of robust, flexible, automated methods to analyze large stacks of images of dense colloidal systems frustrated the ability to extract meaningful and accurate results from one of the experiments without creating new a new analysis tool.

Colloidspy incorporates robust filters, edge detection, and contour description from scikit-image and opencv python packages, and bundles them into  a new class to handle and extract 2D cluster data from colloidal settling images with a minimal learning curve.
It is designed to be used alongside pandas and numpy to control structuring and export of data.

## Installation:

pip install git+https://github.com/UofL-ACE/colloidspy.git#egg=colloidspy


## Example

See example_notebook.ipynb

