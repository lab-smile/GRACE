## GRACE Code Train & Test Tutorial

#### **Intro**
The purpose of this manual is to help you successfully run the GRACE train and test code on the HiperGator through terminal commands and MATLAB GUI on Windows 10 operationg system, which contain the following parts:
- Get the code and the data you need;
- Introduction to the container;
- Run the code by Linux command;
- Check your result;

#### **Prerequisites**
All operations described in this manual are based on a comprehensive enhanced terminal, MobaXterm. You can also use the other cross-platform server application as your preference.

#### **Obatian GRACE code and dataset**

1. Run the MATLAB codes by MATLAB GUI

You can find there are two MATLAB codes, these two MATLAB codes should not require any changes to the path. To run these two MATLAB codes, you need to initiate MATLAB GUI. The MATLAB version that was originally used to run this was R2020b, but it is expected to work with newer MATLAB versions.

You need to select the GRACE working folder and add to path/ Selected Folders and Subfolders firstly before you running these two MATLAB code. Then you can directly run the combine_mask.m without making any change. But if you are using MATLAB R2020b, you need to change line 56 to <center>image(index) = tissue_cond_updated.Labels(k)</center> The output should be a Data folder with the following structure: Data ImagesTr sub-TrX_T1.nii sub-TrXX_T1.nii ... ImagesTs sub-TsX_T1.nii sub-TsXX_T1.nii LabelsTr sub-TrX_seg.nii sub-TrXX_seg.nii LabelsTs sub-TsX_seg.nii sub-TsX_seg.nii<br> Then you need to change your directory to the data folder(/ACT_og_v5) and run makeGRACEjson.m. After this code done, you may exit MATLAB and open terminal to run the other codes.<br>

2. Training and Testing the GRACE code by using terminal

Firstly you need to cd to your own code directory. The GRACE code is using the MONAI, which is an open source foundation. We need to build our own container for MONAI as you build an environment for your code. Run build_container_v08.sh by using the command: <center>sbatch building_container_v08.sh</center>

You need to change the line after sandbox to your desired writable directory and change the line nv to your own directory, I post my own changing as an example:![build_container_example.png](https://s2.loli.net/2022/07/05/7Xv3U8rH1fQcCPu.png)
Then you can find there is a folder named monaicore08 generated under your desired directory.<br>

Now you can train and test the GRACE code. To train the GRACE code, you need to run the runMONAI.sh. Before you training progress, you also need to change the directory. One tips here is that you may change the max_iteration to 100 to see if the code can train successfully. Here is my runMONAI.sh as an example: ![runMONAI.png](https://s2.loli.net/2022/07/05/D9QBP84nVdalGYz.png) For the iterations = 100, the training progress might take about one hours, and for the iterations = 25,000, the training progress might take about 24 hours. Here is another tips, do not forget change the model_save_name to avoid the new model overwritten the old model. Then you can find your training result in the monai_job number.log file which is generated after the training job done.<br>

The test progress is very similar to the training progress. You need to change all paths and make sure the model_save_name matches your model name in runMONAI.sh. Then running the runMONAI_test.sh with sbatch. ![runMONAI_test.png](https://s2.loli.net/2022/07/05/Pk5JwA3fczgG6tR.png)

The outputs for each test subject is saved as a mat file.

3. File Conversion

There is an additional code to convert the .mat files to .nii files under the directory /mat_to_nii. You can run this by adding the list of files that you would like to convert to /mat_to_nii/main.py under the variable "FILES". Then you can run this python code to perform the conversion.

There are also codes to interconvert from .nii to .raw depending on your needs. These files are available in /Nii_Raw_Interconversion as either PYTHON or MATLAB scripts.

4. Visualization

The code for visualizing your results is available at /Visualization Code. Open the file main_v2.py and add your image ID names to the variable SUBLIST. You also need to enter the file paths for each entry in SUBLIST following the example below the variable. You can leave all subject entries as empty quotes ('') other than the T1 and GRACE if you are only running this coding repository. Run main_v2.py after performing these edits.
