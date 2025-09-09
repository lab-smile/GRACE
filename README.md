# Precise and Rapid Whole-Head Segmentation from Magnetic Resonance Images of Older Adults using Deep Learning
We provide open-source code of a pipeline called **General, Rapid, And Comprehensive whole-hEad tissue segmentation**, nicknamed **GRACE**. GRACE is trained and validated on a novel dataset that consists of 177 manually corrected MR-derived reference segmentations that have undergone meticulous manual review. Each T1-weighted MRI volume is segmented into 11 tissue types, including white matter, grey matter, eyes, cerebrospinal fluid, air, blood vessel, cancellous bone, cortical bone, skin, fat, and muscle. GRACE segments a spectrum of tissue types from older adults T1-MRI scans at favorable accuracy and speed. This segmentation only requires the input T1 MRI and does not require special preprocessing in neuroimaging software. The trained GRACE model is optimized on older adult heads to enable high-precision modeling in age-related brain disorders. 

## Paper
This repository provides the official implementation of training GRACE as well as the usage of the model GRACE in the following paper:

**Precise and Rapid Whole-Head Segmentation from Magnetic Resonance Images of Older Adults using Deep Learning**

Skylar E. Stolte<sup>1</sup>, Aprinda Indahlastari<sup>2,3</sup>, Jason Chen<sup>4</sup>, Alejandro Albizu<sup>2,5</sup>, Ayden Dunn<sup>3</sup>, Samantha Pederson<sup>3</sup>, Kyle B. See<sup>1</sup>, Adam J. Woods<sup>2,3,5</sup>, and Ruogu Fang<sup>1,2,6,*</sup>

<sup>1</sup> J. Crayton Pruitt Family Department of Biomedical Engineering, Herbert Wertheim College of Engineering, University of Florida (UF), USA<br>
<sup>2</sup> Center for Cognitive Aging and Memory, McKnight Brain Institute, UF, USA<br>
<sup>3</sup> Department of Clinical and Health Psychology, College of Public Health and Health Professions, UF, USA<br>
<sup>4</sup> Department of Computer & Information Science & Engineering, Herbert Wertheim College of Engineering, University of Florida (UF), USA<br>
<sup>5</sup> Department of Neuroscience, College of Medicine, UF, USA<br>
<sup>6</sup> Department of Electrical and Computer Engineering, Herbert Wertheim College ofEngineering, UF, USA<br>

Imaging NeuroScience<br>
[paper](TBD) | [code](https://github.com/lab-smile/GRACE) | 

## Major results from our work

- Our GRACE segments 11 tissues from T1 MRIs of the human head with high accuracy and fast processing speed.
- GRACE contains its own preprocessing pipeline and does not require the input to be preprocessed in other neuroimaging tools.
- GRACE achieves an average Hausdorff Distance of 0.21, which exceeds the runner-up at an average Hausdorff Distance of 0.36.
- A representative GRACE model is available from this GITHUB. This model may be particularly useful to those who need segmentations of MRIs on older adult heads.

<div align="center">
	<img src="https://github.com/lab-smile/GRACE/blob/main/Images/Figure3.png" width="700">
</div>

<div align="center">
  <b>fig. 1:</b> The segmentation labels for GRACE are as such. These numerical labels correspond to the same labels that appear in the GRACE outputs from this repo. For instance, a voxel output of "1" corresponds to White matter (WM).<br>
</div>
<br>

<div align="center">
	<img src="https://github.com/lab-smile/GRACE/blob/main/Images/Figure11.png" width="700">
</div>

<div align="center">
  <b>fig. 1:</b> Comparison of GRACE with common segmentation methods on four MRI head volumes.<br>
</div>
<br>


## Usage

### Pretrained Models
Our pretrained model can be found at the v1.0.1 release of this GitHub. Please note that there is now two versions: 'GRACE_MONAI081.pth' to work with MONAI 0.8.1 (the original code release from the GRACE paper) and 'GRACE_MONAI150.pth' to work with MONAI 1.5.0 (the most recent MONAI as of September 2025).

### MATLAB Segmentation Label Preparation
You can find there are two MATLAB codes, you can directly change the directory to your own data. You need to select the GRACE working folder and add to path before you running these two MATLAB codes. 

To run the combineIn case of you are using different version of MATLAB, if you are using MATLAB 2020b, you need to change line 56 to :
```
image(index) = tissue_cond_updated.Labels(k)
```
Then you can run the combine_mask.m. The output should be a Data folder with the following structure: 
```
Data ImagesTr sub-TrX_T1.nii sub-TrXX_T1.nii ... 
ImagesTs sub-TsX_T1.nii sub-TsXX_T1.nii ...
LabelsTr sub-TrX_seg.nii sub-TrXX_seg.nii ...
LabelsTs sub-TsX_seg.nii sub-TsX_seg.nii ...
```
Maneuver to the /your_data/Data/. Run make_datalist_json.m

After this code is done, you may exit MATLAB and open the terminal to run the other codes.

### Build container
The GRACE code uses the MONAI, an open-source foundation. We provide a .sh script to help you to build your own container for running your code.

Run the following code in the terminal, you need to change the line after --sandbox to your desired writable directory and change the line after --nv to your own directory.
```
./build_container_v150.sh
```

The output should be a folder named monaicore150 under your desired directory.

### Training
Once the data and the container are ready, you can train the model by using the following command:
```
./train.sh
```
Before you training the model, you need to make sure change the following directory:
- change the first singularity exec -nv to the directory includes monaicore150, for example: /user/GRACE/monaicore150
- change the line after --bind to the directory includes monaicore150
- change the data_dir to your data directory
- change the model name to your desired model name
You can also specify the max iteration number for training. For the iterations = 100, the training progress might take about one hours, and for the iterations = 25,000, the training progress might take about 24 hours. 

### Testing
The test progress is very similar to the training progress. You need to change all paths and make sure the model_save_name matches your model name in runMONAI.sh. Then running the runMONAI_test.sh with the following command: 
```
./test.sh
```
The outputs for each test subject is saved as a mat file.

### File Conversion

There is an additional code to convert the .mat files to .nii files under the directory /mat_to_nii. You can run this by adding the list of files that you would like to convert to /mat_to_nii/main.py under the variable "FILES". Then you can run this python code to perform the conversion. There are also codes to interconvert from .nii to .raw depending on your needs. These files are available in /Nii_Raw_Interconversion as either PYTHON or MATLAB scripts.

### Visualization

The code for visualizing your results is available at /Visualization Code. Open the file main_v2.py and add your image ID names to the variable SUBLIST. You also need to enter the file paths for each entry in SUBLIST following the example below the variable. You can leave all subject entries as empty quotes ('') other than the T1 and GRACE if you are only running this coding repository. Run main_v2.py after performing these edits.


## Citation
If you use this code, please cite our papers:
```
@InProceedings{stolte2024,
  author="Stolte, Skylar E. and Indahlastari, Aprinda and Chen, Jason and Albizu, Alejandro and Dunn, Ayden and Pederson, Samantha and See, Kyle B. and Woods, Adam J. and Fang, Ruogu",
  title="Precise and Rapid Whole-Head Segmentation from Magnetic Resonance Images of Older Adults using Deep Learning",
  booktitle="Imaging NeuroScience",
  year="2024",
  url="TBD"
}
```


## Acknowledgement

This work was supported by the National Institutes of Health/National Institute on Aging (NIA RF1AG071469, NIA R01AG054077), the National Science Foundation (1842473, 1908299, 2123809), the NSF-AFRL INTERN Supplement (2130885), the University of Florida McKnight Brain Institute, the University of Florida Center for Cognitive Aging and Memory, and the McKnight Brain Research Foundation. We acknowledge the NVIDIA AI Technology Center (NVAITC) for their suggestions to this work.


We employ UNETR as our base model from:
https://github.com/Project-MONAI/research-contributions/tree/main/UNETR
## Contact
Any discussion, suggestions and questions please contact: [Skylar Stolte](mailto:skylastolte4444@ufl.edu), [Dr. Ruogu Fang](mailto:ruogu.fang@bme.ufl.edu).

*Smart Medical Informatics Learning & Evaluation Laboratory, Dept. of Biomedical Engineering, University of Florida*
