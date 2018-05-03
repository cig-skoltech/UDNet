function [xFat,xTall] = loadData(imdbPath,fileList,isColor,cid)

if isColor
  imdbPath = fullfile(imdbPath,'color');
else
  imdbPath = fullfile(imdbPath,'gray');
end

fileID = fopen(fileList,'r');
C = strsplit(fscanf(fileID,'%s'),'.jpg');
C(end) = [];
fclose(fileID);

xTall = cast([],cid); % 481 x 321  images
xFat = cast([],cid); % 321 x 481 images

ctr_t = 1;
ctr_f = 1;
for k = 1:numel(C)
  f = single(imread([imdbPath filesep C{k} '.jpg']));
  if size(f,1) > size(f,2)
    xTall(:,:,:,ctr_t) = f;
    ctr_t = ctr_t + 1;
  else
    xFat(:,:,:,ctr_f) = f;
    ctr_f = ctr_f + 1;
  end
end

