%load directory in question (change this to match your file structure)
imgdir = dir('./*/*labelsynced.nii');

%go through directory
for i = 1 : length(imgdir)
    
    %this matches ID in my file names
    file_name = imgdir(i).name(1:end-4);

    %load nii
    imgs = load_untouch_nii(fullfile(imgdir(i).folder, imgdir(i).name)).img;
    
    %get size
    [xx,yy,zz] = size(imgs);
    
    %save as raw type double
    fid=fopen(strcat(file_name, '.raw'),'w+');
    fwrite(fid,imgs,'double');
    fclose(fid);


end

%Test that it worked by opening last raw file:
fid = fopen(strcat(imgdir(i).name(1:end-4),'.raw'));
rawdata = fread(fid, inf, '*double');
rawdata = reshape(rawdata, xx, yy, zz);

figure; imshow(rawdata(:,:,floor(zz/2)), []);