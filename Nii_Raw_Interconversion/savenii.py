import glob
import nibabel as nib
import numpy as np

imgdir = glob.glob('./*/*.raw')
xx, yy, zz = 512, 512, 176

# go through directory
for i in range(len(imgdir)):
    # this matches ID in my file names - might need to change
    file_name = imgdir[i][:-4]

    # open raw file
    with open(imgdir[i], 'rb') as f:
        rawdata = np.fromfile(f, dtype=np.double)
        rawdata = np.reshape(rawdata, (xx, yy, zz))

    nii = nib.Nifti1Image(rawdata, np.eye(4))
    nib.save(nii, imgdir[i][:-4] + '.nii')

    # load back from nii
    imgs = nib.load(imgdir[i][:-4] + '.nii').get_fdata()

    # Test that it worked by opening last raw file:
    plt.imshow(imgs[:, :, int(zz/2)], cmap='gray')
    plt.show()
