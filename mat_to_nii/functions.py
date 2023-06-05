import numpy as np
import scipy.io as sio
import nibabel as nib
import os

def convert_to_nii(files):
    for filepath in files:
        if not os.path.exists(filepath):
            print(f"{filepath} doesn't exist.")
            return False

        newfilepath = ''
        if filepath[-4:] == '.mat':
            newfilepath = filepath[:-4] + '.nii'
        else:
            newfilepath = filepath + '.nii'
        
        # Load the .mat file
        mat_file = sio.loadmat(filepath)

        # Extract the data array from the .mat file
        data = mat_file['testimage']
        data = np.squeeze(data)

        # Create a NIfTI image object and set the data array
        nifti_image = nib.Nifti1Image(data, affine=np.eye(4), dtype='int64')

        # Save the NIfTI image to disk as a .nii file
        nib.save(nifti_image, newfilepath)
        print(f'{filepath} has been converted to {newfilepath}.')
    
    return True