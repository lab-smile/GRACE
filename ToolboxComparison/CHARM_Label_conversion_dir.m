function CHARM_Label_conversion_dir(image_path)
%CHARM labels -> Our labels
%Label Reference
% #   Label Name:
% #   CHARM Label       Our Label
% 0   Background        Background
% 1	  White-Matter		WM	    
% 2   Gray-Matter		GM           
% 3	  CSF			    Eyes	  
% 4	  Bone		        CSF	   	   
% 5	  Scalp			    Air	   
% 6	  Eye_balls			Blood  
% 7	  Compact_bone     	Cancellous Bone	   
% 8	  Spongy_bone     	Cortical Bone	   
% 9   Blood				Skin
% 10  Muscle            Fat
% 11                    Muscle

image_dir = dir(fullfile(image_path, 'm2m*'));

for i = 1 : length(image_dir)

    image_path = fullfile(image_dir(i).folder, image_dir(i).name, 'final_tissues.nii.gz');

    charm = load_untouch_nii(image_path);
    charm_image = charm.img;
    [xx,yy,zz] = size(charm_image);
    
    new_image = zeros(xx,yy,zz);
    
    new_image(charm_image == 1) = 1;
    new_image(charm_image == 2) = 2;
    new_image(charm_image == 6) = 3;
    new_image(charm_image == 3) = 4;
    new_image(charm_image == 9) = 6;
    new_image(charm_image == 8) = 7;
    new_image(charm_image == 7) = 8;
    new_image(charm_image == 5) = 9;
    new_image(charm_image == 10) = 11;
        
    nii = make_nii(new_image);
    nii.hdr = charm.hdr;

    subject_number = image_dir(i).name(5:end);

    save_nii(nii, char(fullfile(image_dir(i).folder, image_dir(i).name, strcat(string(subject_number), '_CHARM_labelsynced.nii'))));
end

end