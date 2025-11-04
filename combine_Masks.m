%% Combine Individual Tissue Masks into a single Tissue Mask
% Created by Alejandro Albizu on 02/02/2022
% Last Updated: 02/02/2022 by AA
clear

%addpath P:\WoodsLab\ACT-head_models\FEM\scripts\utility_codes\NIFTI_20110921\

% rootDir = '/Volumes/woodslab/ACT-head_models/FEM/manual_segmentation/Education_Training';
%rootDir = 'P:\WoodsLab\ACT-head_models\FEM\manual_segmentation\allParticipants\';
rootDir = fullfile("");
%hpgDir = 'W:\camctrp\working\Alejandro\ACT';
% rootDir = 'P:/woodslab/ACT-head_models/FEM/manual_segmentation/Participant_list_v3\idvParticipants';
% rootDir = 'P:\WoodsLab\ACT-head_models\FEM\manual_segmentation\Education_Training';
dims = [256 256 256];
mnames = flip({'wm','gm','eyes','csf','air','blood','cancellous','cortical','skin','fat','muscle'});
idx = flip(1:11); % HARDCODED

%subfdr = dir(fullfile(rootDir,'**','FS*_ses01'));
subfdr = dir(rootDir);
for i = length(subfdr):-1:1
    if ~isdir(fullfile(subfdr(i).folder, subfdr(i).name))
        subfdr(i)=[];
    end
end
subfdr(1:2)=[];

subnames = {subfdr.name}';

tic
missing = zeros(length(subnames),1);
parfor s = 1:length(subnames)
    % if s == 193 || s == 260 || s == 237 || s == 137; continue; end
%     if ~exist(fullfile(rootDir,subnames{s},'T1_T1orT2_masks.nii'),'file')
        masks = zeros(dims);
        maskfdr = dir(fullfile(subfdr(s).folder,subfdr(s).name,'tissueMasks','*.raw')); %'qualityCheck','tissueMasks','*.raw'));
        if length(maskfdr) >= length(mnames)
            for m = 1:length(mnames)
                mask = textscan(fopen(fullfile(maskfdr(1).folder, ...
                    maskfdr(strcmpi(mnames{m},cellfun(@(x) erase(x,'.raw'),...
                    {maskfdr.name},'uni',0))).name)),'%s');
                mask = reshape(double(mask{1}{1}),dims);
                masks(mask ~= 0 & masks ~= 5) = idx(m); % Do Not Allow Tissues Outside of Head
                disp([subnames{s} ': ' mnames{m} ' loaded...']); fclose all;
            end %imagesc(masks(:,:,150),[0 length(maskfdr)])
            
            % PAD CSF
            
            % Preallocate
            allmask = zeros(size(masks));
            gm_mask = masks == idx(strcmp(mnames,'gm')); % HARDCODED
            wm_mask = masks == idx(strcmp(mnames,'wm')); % HARDCODED
            csf_mask = masks == idx(strcmp(mnames,'csf')); % HARDCODED
            bone_mask = ismember(masks, idx(strcmp(mnames,'cancellous') | strcmp(mnames,'cortical'))); % HARDCODED
            
            % Locate GM/bone Intersection
            brain = gm_mask | wm_mask;
            dil_bone = imdilate(bone_mask,ones(3,3,3)); % Dilate bone mask
            dil_bone(bone_mask) = 0; % Dilated portion ONLY of bone mask
            
            % Replace GM touching skull with csf
            masks(gm_mask&dil_bone) = idx(strcmp(mnames,'csf')); % HARDCODED
            
            % SAVE TO PDRIVE
            hdr = load_untouch_header_only( ...
                fullfile(subfdr(s).folder,subnames{s},'T1.nii'));
            nii = make_nii(masks); nii.hdr = hdr;
            save_nii(nii,fullfile(subfdr(s).folder,subnames{s}, ...
                'T1_T1orT2_masks.nii'));

            % COPY TO HIPERGATOR
            %copyfile(fullfile(subfdr(s).folder,subnames{s}, ...
            %    'T1_T1orT2_masks.nii'),...
            %    fullfile(hpgDir,[subnames{s}(1:16) '_ses-01_T1w'], ...
            %    '11tis','T1_T1orT2_masks.nii'))
        else
            missing(s) = 1;
            warning([subnames{s} ': MISSING TISSUE MASKS !!'])
        end
%     end
    disp([subnames{s} ' Complete !'])
end
toc