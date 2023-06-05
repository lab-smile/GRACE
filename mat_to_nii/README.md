# mat_to_nii
Python code to convert MATLAB (mat) files to NIfTI (nii) format.

## Set Up
Download main.py and functions.py.

Other than Python, additional packages may need to be installed to run the program. These include:
* numpy
* nibabel
* scipy

These can all be installed by running the following commands in the command line:

    pip install [name of package]
For example:

    pip install numpy

## Use
Before running the main.py file, be sure to edit the FILES list to include all the filepaths that need to be converted.

    FILES = ['filepath1.mat', 'filepath2.mat']

Note that the files should be .mat files as this code converts them to .nii files.

The resulting files should appear in the same directory as main.py with the same names, except with the .nii file extension instead.