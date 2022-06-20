clc; clear;
ACT_dir = dir('.'); 
for i = length(ACT_dir) : -1 : 1
    if ~isdir(ACT_dir(i).name)
        ACT_dir(i)=[];
    elseif strcmp(ACT_dir(i).name,'.') || strcmp(ACT_dir(i).name,'..')
        ACT_dir(i)=[];
    end
end
names = [];
for i = 1 : length(ACT_dir)
names(i) = str2num(ACT_dir(i).name(11:16));
end
x = 256; y = 256; z = 256;
ccnt=x*y*z;

tissues_cond = readtable('tissue_cond_11.xlsx');
tissue_cond_updated = tissues_cond;
tissue_cond_updated.TissueType(2) = {'wm'};
tissue_cond_updated.TissueType(3) = {'gm'};
tissue_cond_updated.TissueType(8) = {'cancellous'};
tissue_cond_updated.TissueType(9) = {'cortical'};
tissue_cond_updated(13:14,:)=[];

T = length(ACT_dir); Te = round(T*0.10);

mkdir('data')
imagesTr = fullfile('data', 'imagesTr'); mkdir(imagesTr);
imagesTs = fullfile('data', 'imagesTs'); mkdir(imagesTs);
labelsTr = fullfile('data', 'labelsTr'); mkdir(labelsTr);
labelsTs = fullfile('data', 'labelsTs'); mkdir(labelsTs);

for i = 1 : length(ACT_dir)
    path_raw = strcat('./',ACT_dir(i).name,'/idv_mask/');
    path_save = strcat('./',ACT_dir(i).name,'/comb_mask/');

    if ~exist(path_save, 'dir')
       mkdir(path_save)
    end
    
    path_dir = dir(fullfile(path_raw, '*.raw')); 
    image = zeros(x*y*z,1);
    
    for k = height(tissue_cond_updated):-1:1
    
        for j = 1 : length(path_dir)
            fileID = fopen(strcat('/', path_dir(j).folder,'//',path_dir(j).name),'r'); 
            A=fread(fileID,ccnt,'uint8=>uint8'); 
            A=reshape(A,x,y,z);
            fclose(fileID);

            index = find(A==255);

                if contains(path_dir(j).name, string(tissue_cond_updated.TissueType(k)), 'IgnoreCase', true)
                    %labels = tissue_cond_updated.Labels{k} * ones(length(index),1);
                    image(index)=str2num(tissue_cond_updated.Labels{k});

                end

        end
    end
    
    image = reshape(image,x,y,z);
    figure; imshow(image(:,:,floor(z/2)),[0 13]);
    save(strcat(path_save,ACT_dir(i).name(7:16),'_seg.mat'),'image');
    
    seg_file = make_nii(image);
    seg_file.filetype = 2; 
    seg_file.fileprefix = strcat(path_save,ACT_dir(i).name(7:16),'_seg');
    seg_file.machine = 'ieee-le';
    seg_file.original.hdr = seg_file.hdr;
    save_nii(seg_file, char(strcat(path_save,ACT_dir(i).name(7:16),'_seg.nii')));
    
    copyfile(fullfile(ACT_dir(i).name, 'T1.nii'), fullfile(ACT_dir(i).name,strcat(ACT_dir(i).name(7:16), '_T1.nii')));
    
    if (i>(T-Te-1)) && (i<=T)
        copyfile(fullfile(ACT_dir(i).name,'comb_mask', strcat(ACT_dir(i).name(7:16), '_seg.nii')), labelsTs);
        copyfile(fullfile(ACT_dir(i).name,strcat(ACT_dir(i).name(7:16), '_T1.nii')), imagesTs);
    else
        copyfile(fullfile(ACT_dir(i).name,'comb_mask', strcat(ACT_dir(i).name(7:16), '_seg.nii')), labelsTr);
        copyfile(fullfile(ACT_dir(i).name,strcat(ACT_dir(i).name(7:16), '_T1.nii')), imagesTr);
    end
    
end