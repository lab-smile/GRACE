import os
import nibabel as nib
import numpy as np

def CHARM_Label_conversion_dir(image_path):
    # CHARM labels -> Our labels
    # Label Reference
    # #   Label Name:
    # #   CHARM Label       Our Label
    # 0   Background        Background
    # 1   White-Matter      WM
    # 2   Gray-Matter       GM
    # 3   CSF               Eyes
    # 4   Bone              CSF
    # 5   Scalp             Air
    # 6   Eye_balls         Blood
    # 7   Compact_bone      Cancellous Bone
    # 8   Spongy_bone       Cortical Bone
    # 9   Blood             Skin
    # 10  Muscle            Fat
    # 11                    Muscle
    
    image_dir = [f for f in os.listdir(image_path) if f.startswith('m2m')]
    
    for i in range(len(image_dir)):
        image_path = os.path.join(image_path, image_dir[i], 'final_tissues.nii.gz')

        charm = nib.load(image_path)
        charm_image = charm.get_fdata()
        xx, yy, zz = charm_image.shape
        
        new_image = np.zeros((xx, yy, zz))
        
        new_image[charm_image == 1] = 1
        new_image[charm_image == 2] = 2
        new_image[charm_image == 6] = 3
        new_image[charm_image == 3] = 4
        new_image[charm_image == 9] = 6
        new_image[charm_image == 8] = 7
        new_image[charm_image == 7] = 8
        new_image[charm_image == 5] = 9
        new_image[charm_image == 10] = 11
            
        nii = nib.Nifti1Image(new_image, charm.affine, charm.header)
        
        subject_number = image_dir[i][4:]

        nib.save(nii, os.path.join(image_path, f'{subject_number}_CHARM_labelsynced.nii.gz'))
