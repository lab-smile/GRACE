# NIFTI-Visualization
A program created to help visualize different brain segmentation models. Capable of segmenting multiple subjects at once and outputing both outlined and filled in segmentation images. Takes in .nii files and outputs .png images.

## Set Up
This program was created using Python 3.11.1. A link to download the latest version, if needed, can be found [here](https://www.python.org/downloads/).

In addition, this program uses a couple different Python packages, which all need to be installed as well. This can be easily done with the "pip" command, which comes with Python versions 3.4 and later. These packages include:
* numpy
* nibabel
* matplotlib
* Pillow

These can all be installed by doing the following:

    pip install [name of package]

For example:
    
    pip install numpy
## Use
To run the program, only the main.py file needs to be run. Before doing so, however, parts of the main.py file should be changed to create the desired images.

First, the each subject should be added to the SUBLIST dictionary. Each subject should follow this format:

    SUBLIST['subject_name'] = ['T1_filepath', 'charm_filepath', 'headreco_filepath', 'grace_filepath']

If a scan is not needed for a specific subject, its filepath can be replaced with an empty string ('' or "").

    SUBLIST['missing_Grace'] = ['T1_filepath_1', 'charm_filepath_1', 'headreco_filepath_1', '']
NOTE: This program can only take in .nii files.

\
Specific slice views can be changed by editing their respective SLICENUM values.

    AXIAL_SLICENUM = 88

\
The dimensions of the resulting images can be toggled with LENGTH, WIDTH, and HEIGHT. These numbers should be scaled with the original images so the resulting images keep the same aspect ratio.

    LENGTH = 256
    WIDTH = 256
    HEIGHT = 176

\
After these variables have been toggled, the main.py file can be run. When the program is finished, the resulting images will be generated in the same directory as the code files. Each image file name will include the following, each seperated by an underscore:
* The subject's name
* Segmentation type (charm, headreco, or grace)
* Outline, if applicable