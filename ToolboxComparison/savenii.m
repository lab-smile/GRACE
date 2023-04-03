%load directory in question (change this to match your file structure)
imgdir = dir('./*/*.raw');
xx = 512; yy = 512; zz = 176;

%go through directory
for i = 1 : length(imgdir)
    
    %this matches ID in my file names - might need to change
    file_name = imgdir(i).name(1:end-4);
    
    %open raw file
    fid = fopen(fullfile(imgdir(i).folder, imgdir(i).name));
    rawdata = fread(fid, inf, '*double');
    rawdata = reshape(rawdata, xx, yy, zz);

    nii = make_nii(rawdata);
    save_nii(nii, fullfile(imgdir(i).folder, strcat(file_name, '.nii')));

    %load back from nii
    imgs = load_untouch_nii(fullfile(imgdir(i).folder, imgdir(i).name)).img;

    %Test that it worked by opening last raw file:
    figure; imshow(imgs(:,:,floor(zz/2)), []);
    
end