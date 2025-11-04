## 1. Preprocessing the data

This script, preprocess.py, is designed to automate the process of converting a GRACE-style raw dataset into a format compatible with the nnU-Net framework and other medical imaging pipelines.

How to run the file:
```
docker run -v "$(pwd)/data:/data" nikmk26/grace:latest preprocess --source-folders d1 d2 d3 --verbose
```

## 2. Train the model
```
docker run -v "$(pwd)/data:/data" nikmk26/grace:latest train --data_dir /data --model_save_name grace_1 --batch_size_train 1 --batch_size_val 1 --max_iteration 1000 --spatial_size 64 --json_name dataset.json --num_gpu 2
```

## 3. Test the model
```
docker run -v "$(pwd)/data:/data" nikmk26/grace:latest test --data_dir /data --model_load_name grace_1.pth --spatial_size 32 --json_name dataset.json --num_gpu 1
```

## Running Locally with Your Own Changes

If you want to make and test code or configuration changes locally (for example, editing preprocess.py, train.py, or any other file), you can use Docker Compose to rebuild and run everything easily.

### Steps

1. Clone or update the repository
```
git clone https://github.com/lab-smile/GRACE.git
cd GRACE
``` 
But if you already have the repo:
```
git pull
```

2. Make your changes

Edit any file you need — for example:
`preprocess.py`
`train.py`
`docker-compose.yml` or any other source/config file.

3. Rebuild and run the container
```
docker compose up --build
```

This will:
- Rebuild the Docker image with your latest changes.
- Run the container according to the docker-compose.yml configuration (no extra commands needed).


4. Stop the container when finished:
```
docker compose down
```



### Required Data Structure
For the script to run correctly, your data must be organized in a specific hierarchical structure. The script requires a main data directory which contains one or more source folders.

Inside each source folder, there must be individual subject folders. The script identifies subjects by parsing a numeric ID from folder names prefixed with sub- (e.g., sub-12345). Each subject folder must contain the T1-weighted image and its corresponding segmentation mask with specific filenames 


Here is an example of the required layout:

```
/path/to/your/data/      <-- The path provided to the --input argument
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
