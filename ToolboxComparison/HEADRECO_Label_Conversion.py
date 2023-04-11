import os
from pathlib import Path
import numpy as np
import nibabel as nib

def HEADRECO_Label_conversion_dir(image_path):
    # HEADRECO labels -> Our labels
    # Label Reference
    # #   Label Name:
    # #   HEADRECO Label       Our Label
    # 0   Background        Background
    # 1	  WM          		WM	    
    # 2   GM          		GM           
    # 3	  CSF 			    Eyes	  
    # 4	  Bone    		    CSF	   	   
    # 5	  Skin      		Air	   
    # 6	  Eyes          	Blood  
    # 7	                 	Cancellous Bone	   
    # 8	                 	Cortical Bone	   
    # 9     				Skin
    # 10                    Fat
    # 11                    Muscle
    
    image_dir = Path(image_path).glob('m2m*')

    for i, img_dir in enumerate(image_dir, start=1):
        subject_number = img_dir.name[4:]

        image_path = img_dir / (subject_number + '_final_contr.nii.gz')

        headreco = nib.load(str(image_path))
        headreco_image = np.array(headreco.get_fdata())
        xx, yy, zz = headreco_image.shape

        new_image = np.zeros((xx, yy, zz), dtype=np.uint8)

        new_image[headreco_image == 1] = 1
        new_image[headreco_image == 2] = 2
        new_image[headreco_image == 6] = 3
        new_image[headreco_image == 3] = 4
        new_image[headreco_image == 4] = 5
        new_image[headreco_image == 5] = 6

        nii = nib.Nifti1Image(new_image, headreco.affine, header=headreco.header)

        save_path = img_dir / (subject_number + '_HEADRECO_labelsynced.nii')
        nib.save(nii, str(save_path))

HEADRECO_Label_conversion_dir('/path/to/images')
