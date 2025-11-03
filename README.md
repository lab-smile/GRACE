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

### Required Data Structure
The `preprocess.py`, is designed to automate the process of converting a GRACE-style raw dataset into a format compatible with the nnU-Net framework.

For the script to run correctly, your data must be organized in a specific hierarchical structure. The script requires a main data directory which contains one or more source folders. Inside each source folder, there must be individual subject folders. The script identifies subjects by parsing a numeric ID from folder names prefixed with sub- (e.g., sub-12345). Each subject folder must contain the T1-weighted image and its corresponding segmentation mask with specific filenames 

Here is an example of the required layout:

```
/path/to/your/data/      <-- The path provided to the --data_dir argument
├── source_folder_A/     <-- A folder name provided to --source-folders
│   ├── sub-10001/
│   │   ├── T1.nii
│   │   └── T1_T1orT2_masks.nii
│   ├── sub-10002/
│   │   ├── T1.nii
│   │   └── T1_T1orT2_masks.nii
│   └── ...
└── source_folder_B/     <-- Another folder name provided to --source-folders
    ├── sub-30001/
    │   ├── T1.nii
    │   └── T1_T1orT2_masks.nii
    └── ...
```

Preprocessing Workflow
The script performs a three-step preprocessing workflow 

1. File Consolidation: The script first scans the specified --source-folders. It finds all paired T1.nii (image) and T1_T1orT2_masks.nii (label) files. It then copies them into temporary images/ and labels/ directories within your main data path, renaming each file pair using the numeric subject ID (e.g., 10001.nii).
2. Train-Test Split: The consolidated files are split into training and testing sets. The script creates the final nnU-Net style folders: imagesTr, labelsTr, imagesTs, and labelsTs. The split logic is as follows: 
Files are divided into two groups: one for subjects whose IDs start with "1" or "2", and another for subjects whose IDs start with "3".
Within each group, a 90/10 split is performed, allocating 90% of the subjects for the training set (Tr) and 10% for the test set (Ts).
3. JSON Dataset Generation: Finally, the script generates a dataset.json file in the main data directory. This file contains metadata and file paths for the model. Importantly, the script further subdivides the training set (imagesTr, labelsTr) to create a validation set. This final split reserves 10% of the training data for validation, resulting in three distinct sets defined in the JSON: training, validation, and test.

### 1. Singularity Container

#### Singularity Specifics

Our workflow includes several scripts to streamline building the container, preprocessing data, and training or testing the model. Before executing a script, it's helpful to understand the structure of the `singularity` command used to run it.

Here is a complete example command for initiating a python script:
```bash
singularity exec --nv --bind /path/to/working/directory:/mnt /path/to/monai/container/monaicore150.sif python3 /mnt/train.py --flags...
```
This command can be broken down into the following components:

`singularity exec`: The base command to execute a custom script or program inside a specified Singularity container.

`--nv`: A flag that enables NVIDIA GPU support, allowing the container to access and utilize the host system's GPU. Omit this flag if you are running on a CPU-only machine.

`--bind /path/to/working/directory:/mnt`: Mounts a directory from the host machine `(/path/to/working/directory)` to a target directory within the container `(/mnt)`. This is crucial for allowing the container to access your data and scripts, and for ensuring that any output files are saved persistently on the host.

`/path/to/monai/container/monaicore150`: The full path to the Singularity Image File (.sif) you intend to use.

`python3 /mnt/train.py ...`: The actual command that will be run inside the container. This part executes a python script using Python 3, passing along several arguments corresponding to the script, such as the data directory, model name, number of classes, etc.

`!!! Before you run any script make sure to change the mount to your actual input and also include path to your actual monai container.`

#### Build container
The GRACE code uses the MONAI, an open-source foundation. We provide a .sh script to help you to build your own container for running your code.

You can run the script using the following command
```
./build_container_v150.sh
```

The output should be a folder named monaicore150 under your desired directory.

#### Preprocessing
This step is optional if your data is in following structure:
```
data/
├── imagesTr/
│   ├── 111111.nii
│   ├── 111112.nii
│   └── ...
├── imagesTs/
│   ├── 111222.nii
│   ├── 111223.nii
│   └── ...
├── labelsTr/
│   ├── 111111.nii
│   ├── 111112.nii
│   └── ...
├── labelsTs/
│   ├── 111222.nii
│   ├── 111223.nii
│   └── ...
└── dataset.json
```

If not, once the container is ready, make sure your data's folder structure matches the required data structure shown [here](#required-data-structure).

Preprocessing Script Arguments:

| Flag | Description | Required |
| :--- | :--- | :--- |
| `--data` | The path to the main GRACE data directory. | Yes |
| `--source-folders` | A space-separated list of folder names to process within the data directory. | Yes |
| `--verbose` | Enables verbose logging during execution. | No |

You can now run the script with the following command
```
./preprocess.sh
```

The output should be the folder structure as shown above.

#### Training

Training Script Arguments:

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--num_gpu` | Specifies the number of GPUs to use for training. | `3` |
| `--spatial_size` | Sets the patch dimension (height, width, depth) for model input. | `64` |
| `--a_min_value` | Defines the minimum pixel intensity for image normalization. | `0` |
| `--N_classes` | The total number of tissue classes for segmentation. | `12` |
| `--a_max_value` | Defines the maximum pixel intensity for image normalization. | `255` |
| `--max_iteration` | The total number of training iterations to run. | `25000` |
| `--batch_size_train` | Sets the batch size for the training dataset. | `10` |
| `--model_save_name` | The filename for saving the trained model. | `"unetr_v5_cos"` |
| `--batch_size_validation`| Sets the batch size for the validation dataset. | `5` |
| `--json_name` | The name of the JSON file that maps data splits. | `"dataset.json"` |
| `--data_dir` | The directory path where the dataset is located. | `"/red/nvidia-ai/SkylarStolte/training_pairs_v5/"`|

Once the data and the container are ready, you can train the model by using the following command:
```
./train.sh
```

For the iterations = 100, the training progress might take about one hours, and for the iterations = 25,000, the training progress might take about 24 hours. 

Script Outputs

After a training run completes, several files will be saved to the directory specified by `--data_dir`. The filenames are prefixed with the value provided to the `--model_save_name` argument (e.g., "GRACE").

| File Name | Description |
| :--- | :--- |
| `[model_save_name].pth` | A PyTorch model file containing the learned parameters (weights) of the final trained model. |
| `[model_save_name]_Loss.csv` | A CSV file that logs the average training loss at each evaluation interval throughout the training process. |
| `[model_save_name]_training_metrics.pdf`| A PDF document containing two plots: one showing training loss over iterations, and another showing the validation mean Dice score over iterations. |
| `[model_save_name]_ValidationDice.csv` | A CSV file that logs the mean Dice score calculated on the validation dataset at each evaluation interval. |

#### Testing

Testing Script Arguments

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--num_gpu` | Specifies the number of GPUs to use for testing. | `1` |
| `--spatial_size` | The patch dimension (height, width, depth) for the sliding window. | `64` |
| `--a_min_value` | Defines the minimum pixel intensity for image normalization. | `0` |
| `--N_classes` | The total number of tissue classes in the model's output. | `12` |
| `--a_max_value` | Defines the maximum pixel intensity for image normalization. | `255` |
| `--batch_size_test` | Sets the batch size for the testing dataset. | `1` |
| `--model_load_name` | The filename of the trained model (`.pth`) to load for inference. | `"unetr_v5_bfc.pth"`|
| `--dataparallel` | A flag to indicate if the model was trained using multiple GPUs. | `"False"` |
| `--json_name` | The name of the JSON file that maps data splits. | `"dataset.json"` |
| `--data_dir` | The directory path where the dataset is located. | `"/path/to/data/folder/"` |


Once training is done, you can test using the following command: 
```
./test.sh
```
The script generates a segmentation map for each image in the test set. All output files are saved as compressed NIfTI images `(.nii.gz)` inside a new directory at `[data_dir]/TestResults/[model_name]/`. Each output segmentation preserves the header and affine information from its corresponding input image.


### 2. Docker

You can also use docker to run the above given steps, simplest way to do so is via uncommenting the command (i.e. preprocess, train or test) you want to run in the `docker-compose.yml` file and just run `docker compose up --build`. You may also make changes to any other files per your requirement i.e. change any config or file and then proceed to run `docker compose up --build` and when everything is done you may do `docker compose down`.

We also have a published image on Dockerhub, which you can use via following commands:

1. Preprocessing the data
How to run the file:
    ```
    docker run -v "$(pwd)/data:/data" nikmk26/grace:latest preprocess --source-folders d1 d2 d3 --verbose
    ```

2. Train the model
    ```
    docker run -v "$(pwd)/data:/data" nikmk26/grace:latest train --data_dir /data --model_save_name grace_1 --batch_size_train 1 --batch_size_val 1 --max_iteration 1000 --spatial_size 64 --json_name dataset.json --num_gpu 2
    ```

3. Test the model
    ```
    docker run -v "$(pwd)/data:/data" nikmk26/grace:latest test --data_dir /data --model_load_name grace_1.pth --spatial_size 32 --json_name dataset.json --num_gpu 1
    ```

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
