import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# set directory path
directory = "./"

# get all files ending in '*labelsynced.nii' in the directory and its subdirectories
imgdir = [os.path.join(root, name) for root, dirs, files in os.walk(directory) for name in files if name.endswith('*labelsynced.nii')]

# go through directory
for i in range(len(imgdir)):
    # this matches ID in my file names
    file_name = os.path.splitext(os.path.basename(imgdir[i]))[0]

    # load nii
    imgs = nib.load(imgdir[i]).get_fdata()

    # get size
    xx, yy, zz = imgs.shape

    # save as raw type double
    with open(file_name + '.raw', 'wb') as f:
        f.write(imgs.astype(np.float64).tobytes())

    # test that it worked by opening last raw file
    with open(file_name + '.raw', 'rb') as f:
        rawdata = np.frombuffer(f.read(), dtype=np.float64)
        rawdata = rawdata.reshape(xx, yy, zz)

    plt.imshow(rawdata[:, :, zz//2], cmap='gray')
    plt.show()
