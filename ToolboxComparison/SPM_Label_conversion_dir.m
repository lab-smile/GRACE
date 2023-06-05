function SPM_Label_conversion_dir(image_path)
%HEADRECO labels -> Our labels
%Label Reference
% #   Label Name:
% #   HEADRECO Label       Our Label
% 0   Background        Background          0
% 1	  GM          		WM	                2
% 2   WM          		GM                  1
% 3	  CSF 			    Eyes	            3
% 4	  Bone    		    CSF	   	            3
% 5	  Skin      		Air	                0
% 6	  Air          	    Blood               3
% 7	                 	Cancellous Bone	    4
% 8	                 	Cortical Bone	    4
% 9     				Skin                5
% 10                    Fat                 5
% 11                    Muscle              5

image_dir = dir(image_path);
for j = length(image_dir):-1:1
    if ~isdir(image_dir(j).name) || contains(image_dir(j).name, '.') || contains(image_dir(j).name, 'spm')
        image_dir(j) = [];
    end
end

for i = 1 : length(image_dir)

    subject_number = image_dir(i).name;
%     image_sub_dir = dir(image_dir(i).name);

    image_path = fullfile(image_dir(i).folder, image_dir(i).name, strcat('sub-', subject_number, '_T1_flirt.nii'));

    T1 = load_untouch_nii(image_path);
    T1_image = T1.img;
    [xx,yy,zz] = size(T1_image);
    
%     new_image = zeros(xx,yy,zz);

    c1 = load_untouch_nii(fullfile(image_dir(i).folder, image_dir(i).name, strcat('c1sub-', subject_number, '_T1_flirt.nii'))).img;
    c2 = load_untouch_nii(fullfile(image_dir(i).folder, image_dir(i).name, strcat('c2sub-', subject_number, '_T1_flirt.nii'))).img;
    c3 = load_untouch_nii(fullfile(image_dir(i).folder, image_dir(i).name, strcat('c3sub-', subject_number, '_T1_flirt.nii'))).img;
    c4 = load_untouch_nii(fullfile(image_dir(i).folder, image_dir(i).name, strcat('c4sub-', subject_number, '_T1_flirt.nii'))).img;
    c5 = load_untouch_nii(fullfile(image_dir(i).folder, image_dir(i).name, strcat('c5sub-', subject_number, '_T1_flirt.nii'))).img;
    c6 = load_untouch_nii(fullfile(image_dir(i).folder, image_dir(i).name, strcat('c6sub-', subject_number, '_T1_flirt.nii'))).img;

    c1 = c1/255; c2 = c2/255; c3 = c3/255; c4 = c4/255; c5 = c5/255; c6 = c6/255;
    image_mapped = zeros(xx,yy,zz,6);

    image_mapped(:,:,:,1) = c6; 
    image_mapped(:,:,:,2) = c2; 
    image_mapped(:,:,:,3) = c1; 
    image_mapped(:,:,:,4) = c3; 
    image_mapped(:,:,:,5) = c4; 
    image_mapped(:,:,:,6) = c5; 

    [max_val, max_idx] = max(image_mapped, [], 4);
        
    nii = make_nii(max_idx);
    nii.hdr = T1.hdr;

    save_nii(nii, char(fullfile('spm', strcat(string(subject_number), '_spm_5label.nii'))));
end

end