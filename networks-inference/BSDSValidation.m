    function results = BSDSValidation(varargin)
% Runs the two networks for all the images in the BSDS68 validation dataset

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_layers', 'vl_setupnn.m'));

opts.patchSize = 7;
opts.randn_seed = 19092015;
opts.color = false;
opts.noise_std = 5:5:55;
opts.batchSize = 25;
opts.gpus = [];
opts.cid = 'single';
opts.oracle = true;
opts.savePath = fullfile(fileparts(mfilename('fullpath')),['Results_',datestr(now,'yyyy-mm-dd_HHMMSSFFF')]);
opts.imdbPath = fullfile(fileparts(mfilename('fullpath')),'..','datasets','BSDS500');
opts.fileList = 'BSDS_validation_list.txt';
opts.methods = {'unet','unlnet'};


opts = vl_argparse(opts,varargin);

if opts.color, opts.patchSize = 5; type = 'color'; else, type = 'gray'; end

results = struct('Obs',[]);
for k = 1:numel(opts.methods)
  results.(opts.methods{k}) = [];
  if opts.oracle && ~isempty(intersect(opts.methods{k},{'unlnet'}))
    results.([opts.methods{k} '_oracle'])=[];
  end
end

if isempty(opts.savePath)
  opts.savePath = fullfile(fileparts(mfilename('fullpath')),'Results');
end
if ~exist(opts.savePath, 'dir')
  mkdir(opts.savePath);
end

fprintf('Loading data ...\n');
[xF,xT] = loadData(opts.imdbPath,opts.fileList,opts.color,opts.cid);
fprintf('Data were loaded ...\n');

numFimages = size(xF,4);
numTimages = size(xT,4);

useGPU = ~isempty(opts.gpus);

for k = 1:numel(opts.noise_std)
  fprintf('*********************************************\n');
  fprintf('Computing results for noise std:%d\n',opts.noise_std(k));
  
  if useGPU
    fprintf('Initializing gpu %d ...\n',opts.gpus(1))
    clearMex();
    gpuDevice(opts.gpus(1))
  end
  
  % Initialize the seed for the random generator
  s = RandStream('mt19937ar','Seed',opts.randn_seed);
  RandStream.setGlobalStream(s);
  
  stdn = opts.noise_std(k);
  yF = xF + stdn*randn(size(xF),'like',xF);
  yT = xT + stdn*randn(size(xT),'like',xT);
  
  psnr = zeros(numTimages+numFimages,1,opts.cid);
  for m=1:numTimages
    psnr(m) = misc.psnr(yT(:,:,:,m),xT(:,:,:,m),255);
  end
  for m=numTimages+1:numTimages+numFimages
    d = m-numTimages;
    psnr(m) = misc.psnr(yF(:,:,:,d),xF(:,:,:,d),255);
  end
  
  if k == 1
    results.Obs = struct('type',type,'std',stdn,'psnr',psnr, ...
      'avg_psnr',mean(psnr),'seed',opts.randn_seed);
  else
    results.Obs(k) = struct('type',type,'std',stdn,'psnr',psnr, ...
      'avg_psnr',mean(psnr),'seed',opts.randn_seed);
  end
  
  
  if ~isempty(intersect(opts.methods,{'unlnet'}))
    
    fprintf('Computing block-matching ...\n');
    
    if useGPU
      [yT,yF] = misc.move_data('gpu',yT,yF);
    end
    
    idxT = misc.patchMatch(nn_pad(yT,floor(opts.patchSize/2)*ones(1,4)), ...
      'stride',[1 1],'Nbrs',8,'searchwin',[15 15],'patchsize',ones(1,2)*opts.patchSize);
    
    idxF = misc.patchMatch(nn_pad(yF,floor(opts.patchSize/2)*ones(1,4)), ...
      'stride',[1 1],'Nbrs',8,'searchwin',[15 15],'patchsize',ones(1,2)*opts.patchSize);
    
    idxT = gather(idxT);
    idxF = gather(idxF);
        
    if useGPU
      [yT,yF] = misc.move_data('cpu',yT,yF);
    end    
    
    if opts.oracle && k==1
      
      if useGPU
        [xT,xF] = misc.move_data('gpu',xT,xF);
      end
      
      idxT_orc = misc.patchMatch(nn_pad(xT,floor(opts.patchSize/2)*ones(1,4)), ...
        'stride',[1 1],'Nbrs',8,'searchwin',[15 15],'patchsize',ones(1,2)*opts.patchSize);
      
      idxF_orc = misc.patchMatch(nn_pad(xF,floor(opts.patchSize/2)*ones(1,4)), ...
        'stride',[1 1],'Nbrs',8,'searchwin',[15 15],'patchsize',ones(1,2)*opts.patchSize);
      
      idxT_orc = gather(idxT_orc);
      idxF_orc = gather(idxF_orc);
    
      if useGPU
        [xT,xF] = misc.move_data('cpu',xT,xF);
      end
      
    end
    fprintf('Block-matching computation has finished.\n');
  end
  
  for l = 1:numel(opts.methods)
    fprintf('Computing results for method: %s\n',opts.methods{l});
    switch opts.methods{l}
      case 'unlnet'
        
        xe_F = zeros(size(yF),opts.cid);
        N = size(yF,4);
        for t = 1:opts.batchSize:N
          batchStart = t;
          batchEnd = min(t+opts.batchSize-1,N);
          if useGPU
            y = gpuArray(yF(:,:,:,batchStart:batchEnd));
            idx = gpuArray(idxF(:,:,:,batchStart:batchEnd));
          else
            y = yF(:,:,:,batchStart:batchEnd);
            idx = idxF(:,:,:,batchStart:batchEnd);
          end
          out = unlnet_den(y,stdn,'idx',idx,'fsize',opts.patchSize);
          xe_F(:,:,:,batchStart:batchEnd) = gather(out);
        end
        clear out y idx;        
        
        xe_T = zeros(size(yT),opts.cid);
        N = size(yT,4);
        for t = 1:opts.batchSize:N
          batchStart = t;
          batchEnd = min(t+opts.batchSize-1,N);
          if useGPU
            y = gpuArray(yT(:,:,:,batchStart:batchEnd));
            idx = gpuArray(idxT(:,:,:,batchStart:batchEnd));
          else
            y = yT(:,:,:,batchStart:batchEnd);
            idx = idxT(:,:,:,batchStart:batchEnd);
          end
          out = unlnet_den(y,stdn,'idx',idx,'fsize',opts.patchSize);
          xe_T(:,:,:,batchStart:batchEnd) = gather(out);
        end
        clear out y idx;
                        
        psnr = zeros(numTimages+numFimages,1,opts.cid);
        for m=1:numTimages
          psnr(m) = misc.psnr(xe_T(:,:,:,m),xT(:,:,:,m),255);
        end
        for m=numTimages+1:numTimages+numFimages
          d = m-numTimages;
          psnr(m) = misc.psnr(xe_F(:,:,:,d),xF(:,:,:,d),255);
        end
        
        if k == 1
          results.unlnet = struct('type',type,'std',stdn,'patchSize', ...
            opts.patchSize,'psnr',psnr,'avg_psnr',mean(psnr), ...
            'seed',opts.randn_seed);
        else
          results.unlnet(k) = struct('type',type,'std',stdn,'patchSize', ...
            opts.patchSize,'psnr',psnr,'avg_psnr',mean(psnr), ...
            'seed',opts.randn_seed);
        end
        
        if opts.oracle
          
          xe_F = zeros(size(yF),opts.cid);
          N = size(yF,4);
          for t = 1:opts.batchSize:N
            batchStart = t;
            batchEnd = min(t+opts.batchSize-1,N);
            if useGPU
              y = gpuArray(yF(:,:,:,batchStart:batchEnd));
              idx = gpuArray(idxF_orc(:,:,:,batchStart:batchEnd));
            else
              y = yF(:,:,:,batchStart:batchEnd);
              idx = idxF_orc(:,:,:,batchStart:batchEnd);
            end
            out = unlnet_den(y,stdn,'idx',idx,'fsize',opts.patchSize);
            xe_F(:,:,:,batchStart:batchEnd) = gather(out);
          end
          clear out y idx;
          
          xe_T = zeros(size(yT),opts.cid);
          N = size(yT,4);
          for t = 1:opts.batchSize:N
            batchStart = t;
            batchEnd = min(t+opts.batchSize-1,N);
            if useGPU
              y = gpuArray(yT(:,:,:,batchStart:batchEnd));
              idx = gpuArray(idxT_orc(:,:,:,batchStart:batchEnd));
            else
              y = yT(:,:,:,batchStart:batchEnd);
              idx = idxT_orc(:,:,:,batchStart:batchEnd);
            end
            out = unlnet_den(y,stdn,'idx',idx,'fsize',opts.patchSize);
            xe_T(:,:,:,batchStart:batchEnd) = gather(out);
          end
          clear out y idx;
          
          psnr = zeros(numTimages+numFimages,1,opts.cid);
          for m=1:numTimages
            psnr(m) = misc.psnr(xe_T(:,:,:,m),xT(:,:,:,m),255);
          end
          for m=numTimages+1:numTimages+numFimages
            d = m-numTimages;
            psnr(m) = misc.psnr(xe_F(:,:,:,d),xF(:,:,:,d),255);
          end
          
          if k == 1
            results.unlnet_oracle = struct('type',type,'std',stdn, ...
              'patchSize', opts.patchSize,'psnr',psnr,'avg_psnr', ...
              mean(psnr),'seed',opts.randn_seed);
          else
            results.unlnet_oracle(k) = struct('type',type,'std',stdn, ...
              'patchSize', opts.patchSize,'psnr',psnr,'avg_psnr', ...
              mean(psnr),'seed',opts.randn_seed);
          end
          
        end
        
      case 'unet'
        
        xe_F = zeros(size(yF),opts.cid);
        N = size(yF,4);
        for t = 1:opts.batchSize:N
          batchStart = t;
          batchEnd = min(t+opts.batchSize-1,N);
          if useGPU
            y = gpuArray(yF(:,:,:,batchStart:batchEnd));          
          else
            y = yF(:,:,:,batchStart:batchEnd);           
          end
          out = unet_den(y,stdn,'fsize',opts.patchSize);
          xe_F(:,:,:,batchStart:batchEnd) = gather(out);
        end
        clear out y;
        
        xe_T = zeros(size(yT),opts.cid);
        N = size(yT,4);
        for t = 1:opts.batchSize:N
          batchStart = t;
          batchEnd = min(t+opts.batchSize-1,N);
          if useGPU
            y = gpuArray(yT(:,:,:,batchStart:batchEnd));            
          else
            y = yT(:,:,:,batchStart:batchEnd);            
          end
          out = unet_den(y,stdn,'fsize',opts.patchSize);
          xe_T(:,:,:,batchStart:batchEnd) = gather(out);
        end
        clear out y;          
                
        psnr = zeros(numTimages+numFimages,1,opts.cid);
        for m=1:numTimages
          psnr(m) = misc.psnr(xe_T(:,:,:,m),xT(:,:,:,m),255);
        end
        for m=numTimages+1:numTimages+numFimages
          d = m-numTimages;
          psnr(m) = misc.psnr(xe_F(:,:,:,d),xF(:,:,:,d),255);
        end
        
        if k == 1
          results.unet = struct('type',type,'std',stdn,'patchSize', ...
            opts.patchSize,'psnr',psnr,'avg_psnr',mean(psnr), ...
            'seed',opts.randn_seed);
        else
          results.unet(k) = struct('type',type,'std',stdn,'patchSize', ...
            opts.patchSize,'psnr',psnr,'avg_psnr',mean(psnr), ...
            'seed',opts.randn_seed);
        end
    end
    fprintf('Results for method: %s are complete.\n',opts.methods{l});    
  end
  fprintf('Results for noise std:%d are complete.\n',opts.noise_std(k));
  fprintf('*********************************************\n');
end

if opts.color
  save(fullfile(opts.savePath, 'res_color.mat'),'results');
else
  save(fullfile(opts.savePath, 'res_gray.mat'),'results');
end


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


% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
%clear vl_tmove vl_imreadjpeg ;
disp('Clearing mex files');
clear mex;
clear vl_tmove vl_imreadjpeg;
