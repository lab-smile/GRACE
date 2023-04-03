function HEADRECO_Label_conversion_dir(image_path)
%HEADRECO labels -> Our labels
%Label Reference
% #   Label Name:
% #   HEADRECO Label       Our Label
% 0   Background        Background
% 1	  WM          		WM	    
% 2   GM          		GM           
% 3	  CSF 			    Eyes	  
% 4	  Bone    		    CSF	   	   
% 5	  Skin      		Air	   
% 6	  Eyes          	Blood  
% 7	                 	Cancellous Bone	   
% 8	                 	Cortical Bone	   
% 9     				Skin
% 10                    Fat
% 11                    Muscle

image_dir = dir(fullfile(image_path, 'm2m*'));

for i = 1 : length(image_dir)

    subject_number = image_dir(i).name(5:end);

    image_path = fullfile(image_dir(i).folder, image_dir(i).name, strcat(subject_number, '_final_contr.nii.gz'));

    headreco = load_untouch_nii(image_path);
    headreco_image = headreco.img;
    [xx,yy,zz] = size(headreco_image);
    
    new_image = zeros(xx,yy,zz);
    
    new_image(headreco_image == 1) = 1;
    new_image(headreco_image == 2) = 2;
    new_image(headreco_image == 6) = 3;
    new_image(headreco_image == 3) = 4;
    new_image(headreco_image == 4) = 5;
    new_image(headreco_image == 5) = 6;
        
    nii = make_nii(new_image);
    nii.hdr = headreco.hdr;

    save_nii(nii, char(fullfile(image_dir(i).folder, image_dir(i).name, strcat(string(subject_number), '_HEADRECO_labelsynced.nii'))));
end

end