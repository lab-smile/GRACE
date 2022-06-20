% fname = 'dataset_2.json'; 
% fid = fopen(fname); 
% raw = fread(fid,inf); 
% str = char(raw'); 
% fclose(fid); 
% val = jsondecode(str);

description = 'AISEG V5 - Code Validation';
labels = struct('xO', 'background', ...
    'x1', 'wm', ...
    'x2', 'gm', ...
    'x3', 'eyes', ...
    'x4', 'csf', ...
    'x5', 'air', ...
    'x6', 'blood', ...
    'x7', 'cancellous', ...
    'x8', 'cortical', ...
    'x9', 'skin', ...
    'x10', 'fat', ...
    'x11', 'muscle');
license = 'UF';
modality = struct('x0', 'T1');
numTest = length(dir(fullfile('imagesTs', '*.nii')));
numTraining = length(dir(fullfile('imagesTr', '*.nii')));

testdir = dir(fullfile('imagesTs', '*.nii'));
for i = 1 : length(testdir)
    test(i,1) = {fullfile('imagesTs',testdir(i).name)};
end

traindir = dir(fullfile('imagesTr', '*.nii'));
trainlabeldir = dir(fullfile('labelsTr', '*.nii'));

T = length(traindir); Te = round(T*0.10);
traindir_fin = traindir(1:T-Te-1);
validdir_fin = traindir(T-Te:T);
traindir_label_fin = trainlabeldir(1:T-Te-1);
validdir_label_fin = trainlabeldir(T-Te:T);

%training = struct('image', fullfile('imagesTr',traindir_fin(:).name), 'label', fullfile('labelsTr',traindir_label_fin(:).name));
% training = struct();
% training.image = {fullfile('imagesTr',traindir_fin(:).name)};
% training.label = {fullfile('labelsTr',traindir_label_fin(:).name)};
% training = struct('image', fullfile('imagesTr',traindir_fin(:).name), 'label', fullfile('labelsTr',traindir_label_fin(:).name));
% training = [image, label];
for i = 1 : length(traindir_fin)
    newtrain = struct('image', fullfile('imagesTr',traindir_fin(i).name), 'label', fullfile('labelsTr',traindir_label_fin(i).name));
    if i == 1
        training = newtrain;
    else
        training = [training; newtrain];
    end
end
for i = 1 : length(validdir_fin)
    newtrain = struct('image', fullfile('imagesTr',validdir_fin(i).name), 'label', fullfile('labelsTr',validdir_label_fin(i).name));
    if i == 1
        validation = newtrain;
    else
        validation = [validation; newtrain];
    end
end

% s = struct('description', description, ...
%     'license', license, ...
%     'labels', labels, ...
%     'modality', modality, ...
%     'name', 'ACT', ...
%     'numTest', numTest, ...
%     'numTraining', numTraining, ...
%     'reference', 'NA', ...
%     'release', 'NA', ...
%     'tensorImageSize', '3D', ...
%     'test', test, ...
%     'training', training, ...
%     'validation', validation);
s = struct();
s.description = description;
s.labels = labels;
s.license = license;
s.modality = modality;
s.name = 'ACT';
s.numTest = numTest;
s.numTraining = numTraining;
s.reference = 'NA';
s.release = 'NA';
s.tensorImageSize = '3D';
s.test = test;
s.training = training;
s.validation = validation;
    
s = jsonencode(s);
fid = fopen('dataset_1.json','w');
fprintf(fid,s);
fclose(fid);
    
    
    