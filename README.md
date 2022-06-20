# GRACE

#Steps for Running:

1. Assume the subject data subdirectories are in here. The two MATLAB codes should not require any changes to the path.
2. Initiate MATLAB GUI. I always use https://ood.rc.ufl.edu.
3. Run combine_masks.m. The output should be a Data folder with the following structure:
	
	Data
		ImagesTr
			sub-TrX_T1.nii
			sub-TrXX_T1.nii
			...
		ImagesTs
			sub-TsX_T1.nii
			sub-TsXX_T1.nii
		LabelsTr
			sub-TrX_seg.nii
			sub-TrXX_seg.nii
		LabelsTs
			sub-TsX_seg.nii
			sub-TsX_seg.nii

4. Maneuver to the Data folder. Run makeGRACEjson.m

5. Open build_container_v08.sh and change the paths after "--sandbox" and "--nv" to your own desired container location. Run the sh file using sbatch.

6. Open runMONAI.sh. Change all path locations (and model_save_name if desired). Run the sh file suing sbatch.

7. Confirm that the trained model, loss and dice curves, and loss and dice csvs have appeared in the Data folder upon training completion.

8. Open runMONAI_test.sh and change all paths (and ensure model_save_name matches your model from step 6). Run with sbatch.