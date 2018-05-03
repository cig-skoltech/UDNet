function net = unlnet_joint_train(net,varargin)

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_layers', 'vl_setupnn.m')) ;

%  ---- Network Training Parameters ------
opts.numEpochs = 300; % Maximum number of gradient-descent iterations
opts.batchSize = 50;
opts.batchSize_val = 100;
opts.learningRate = 0.001;
opts.solver = @solver.momentum;
opts.solverOpts = [];%struct('gamma', 0.9, 'decay', 0);
opts.cudnn = true;
opts.backPropDepth = +inf;
opts.net_move = @net_move;
opts.net_eval = @unlnet_eval;
opts.patchFile = []; % Mat file that contains the Nbrs_idx and dist 
% for all the training and testing data.

%Indicate whether to use or not a gpu for the training.
opts.gpus = 1; %[] -> no gpu
opts.plotStatistics = false; % if set to true the graph with the objective is ploted
opts.saveFreq = 10; % Results are saved every opts.saveFreq number of epochs.
opts.conserveMemory = true;

opts.imdbPath = fullfile('..','data', 'imdb.mat');

opts.net_struct={...
  struct('layer_type','unlnet','learningRate',1,'first_stage',true), ...
  struct('layer_type','unlnet','learningRate',1,'first_stage',false), ...
  struct('layer_type','unlnet','learningRate',1,'first_stage',false), ...
  struct('layer_type','unlnet','learningRate',1,'first_stage',false), ...
  struct('layer_type','unlnet','learningRate',1,'first_stage',false), ...
  struct('layer_type','clip','lb',0,'ub',255),...  
  struct('layer_type','imloss','peakVal',255)};

opts.noise_std = [5,9,13,17,21,25,29];
%  ------ Inverse Problem Parameters ------
% Define the forward model of the inverse problem we are training the
% network to solve.
% y=A(x)+n where A is the forward operator and n is gaussian noise of
% noise_std.

opts.name_id = 'unlnet';
opts.randn_seed = 124324;

% layer-specific paramaters
%  ----- dprNet -------
opts.patchSize = [5 5];
opts.numFilters = [];
opts.stride = [1,1];
opts.padSize = [];
opts.padType = 'symmetric';
opts.weightSharing = false;
opts.groupWeightSharing = false;
opts.zeroMeanFilters = true;
opts.weightNormalization = true(1,2);
opts.rbf_means = -100:4:100;
opts.rbf_precision = [];
opts.rbf_weights = [];
opts.h = [];
opts.ht = [];
opts.s = [];
opts.st = [];
opts.g = [];
opts.gt = [];
opts.lr = ones(1,8); % Learning rate for each weight of the stage
opts.first_stage = true;
opts.data_mu = [];
opts.step = 0.1;
opts.origin = -104;
opts.shrink_type = 'identity';
opts.alpha = 0;
opts.clb = -100;
opts.cub = 100;

% Patch-Match parameters
opts.searchwin = [15 15];
opts.useSep = true;
opts.transform = false;
opts.patchDist = 'euclidean';
opts.sorted = true;
opts.Wx = [];
opts.Wy = [];
opts.Nbrs = 8;
opts.DoubleNoiseRealization = false;
opts.multichannel_dist = false; % If set to true the computation of the 
% distance between patches is performed using the multichannel patch. 
% Otherwise it is computed using a single channel patch, which is derived 
% as the mean of the image channels. 

% --- Imloss layer -----
opts.peakVal = 255;
opts.loss_type = 'psnr';

% --- CLIP layer -----
opts.lb = 0;
opts.ub = 255;

opts.cid = 'single';
%opts.cid='double';

[opts,varargin] = vl_argparse(opts, varargin);

if isempty(opts.solverOpts)
  opts.solverOpts = opts.solver();
end

if isempty(opts.data_mu)
  opts.data_mu=cast(opts.origin:opts.step:-opts.origin,opts.cid);
  opts.data_mu=bsxfun(@minus,opts.data_mu,cast(opts.rbf_means(:),opts.cid));
end

opts.netParams = struct('data_mu',opts.data_mu,'step',opts.step,'origin',opts.origin);

% How many unlnet stages the network consists of.
if isempty(net)  
  numStages = 0;
  for k=1:numel(opts.net_struct)
    if isequal(opts.net_struct{k}.layer_type,'unlnet')
      numStages = numStages+1;
    end
  end
else
  numStages = 0;
  for k=1:numel(net.layers)
    if isequal(net.layers{k}.type,'unlnet')
      numStages = numStages+1;
    end
  end
end

str_solver = char(opts.solver);
str_solver = str_solver(strfind(str_solver,'.')+1:end);

if numel(opts.noise_std) == 1
  str_std = sprintf('%0.f',opts.noise_std(1));
else
  str_std = [ '[' num2str(opts.noise_std) ']' ];
  str_std = regexprep(str_std,'\s*',',');
end

if opts.weightSharing
  str_ws = '-WS';
else
  str_ws = '';
end

if opts.groupWeightSharing
  str_gs = '-GS';
else
  str_gs = '';
end

opts.expDir = fullfile('Results', ...
  sprintf('%s-stages:%.0f-psize:%.0fx%.0f@%d%s%s-std:%s-solver:%s-jointTrain',...
  opts.name_id,numStages,opts.patchSize(1),opts.patchSize(2),...
  opts.numFilters,str_ws,str_gs,str_std,str_solver));

if ~exist(opts.expDir, 'dir')
  mkdir(opts.expDir);
  copyfile([mfilename '.m'], [opts.expDir filesep]);
  save([opts.expDir filesep 'arg_In'],'opts');
end

opts = vl_argparse(opts, varargin);

% -------------------------------------------------------------------------
%                Prepare Data and Model
% -------------------------------------------------------------------------
imdb = load(opts.imdbPath);
if ~isequal(opts.cid,'single')
  imdb.images.data = cast(imdb.images.data,opts.cid);
end

 imdb.images.data = imdb.images.data(:,:,:,313:321);% 8:10
 imdb.images.set = imdb.images.set(313:321);

noise_levels = numel(opts.noise_std);
imdb.images.set = imdb.images.set(:);
imdb.images.set = repmat(imdb.images.set,noise_levels,1); % Every image is 
% corrupted by K different noise levels and all the K instances are used 
% either for training or for validation.


% Create the noise added to the data according to the chosen standard
% deviation of the noise.

% Initialize the seed for the random generator
s = RandStream('mt19937ar','Seed',opts.randn_seed);
RandStream.setGlobalStream(s);

% The degraded input that we feed to the network and we want to
% reconstruct.
imdb.images.obs = [];
for k=1:noise_levels
  imdb.images.obs = cat(4,imdb.images.obs, imdb.images.data + ...
  opts.noise_std(k)*randn(size(imdb.images.data),opts.cid));
end
imdb.images.noise_std = opts.noise_std;
imdb.images.stage_input = [];

opts.inputSize = size(imdb.images.data(:,:,:,1));
% Initialize network parameters
% Initialize network parameters
if isempty(net)
  net = net_init_from_struct(opts);
else
  net.meta.trainOpts.inputSize = opts.inputSize;
  net.meta.trainOpts.noise_std = opts.noise_std;
  net.meta.trainOpts.randSeed = opts.randn_seed;
  net.meta.trainOpts.numEpochs = opts.numEpochs;
  net.meta.trainOpts.learningRate = opts.learningRate;
  net.meta.trainOpts.optimizer = char(opts.solver);
  net.meta.netParams = opts.netParams;
end

% -------------------------------------------------------------------------
%                     Train Network Stage by Stage
% -------------------------------------------------------------------------

train_image_set = find(imdb.images.set == 1);
val_image_set = find(imdb.images.set == 2);

% -------------------------------------------------------------------------
% Compute patch-similarity based either on the noisy input or in a initial
% denoised estimate and use the same similarity indices for all stages.
% -------------------------------------------------------------------------
if isempty(opts.patchFile)
  opts.patchFile = fullfile(opts.expDir,'patchSimilarityIndices.mat');
end
opts_PM = struct('searchwin',opts.searchwin,'useSep',opts.useSep, ...
  'transform',opts.transform,'patchDist',opts.patchDist,'sorted', ...
  opts.sorted,'Wx',opts.Wx,'Wy',opts.Wy,'Nbrs',opts.Nbrs,'cudnn', ...
  opts.cudnn, 'padSize', net.layers{1}.padSize, 'padType', ...
  net.layers{1}.padType,'gpus',opts.gpus,'patchSize', ...
  size(net.layers{1}.weights{1}(:,:,1,1)),'stride',net.layers{1}.stride, ...
  'batchSize',opts.batchSize_val,'multichannel_dist', opts.multichannel_dist);

if exist(opts.patchFile, 'file')
   load(opts.patchFile,'Nbrs_idx');
else
  if opts.DoubleNoiseRealization
    % We find the similar patches not from the input of the network but
    % from the same underlying images but degraded from a different noise
    % realization. 
    obs = [];
    for k=1:noise_levels
      obs = cat(4,obs, imdb.images.data + ...
        opts.noise_std(k)*randn(size(imdb.images.data),opts.cid));
    end
    % The validation data will have the same noise realization as the input
    % of the network.
    obs(:,:,:,val_image_set) = imdb.images.obs(:,:,:,val_image_set);    
  else
    obs =  imdb.images.obs;
  end
  
  [Nbrs_idx,Nbrs_dist] = precompute_knn_idx(obs,opts_PM); %#ok<ASGLU>
  clear obs;
  save(opts.patchFile,'Nbrs_idx','Nbrs_dist','-v7.3'); 
  clear Nbrs_dist;
end


start_time = tic;
net = deep_net_train(net, imdb, @(x,y)getBatch(x,y,Nbrs_idx), ...
  'expDir', opts.expDir, ...
  'solver', opts.solver, ...
  'solverOpts', opts.solverOpts, ...
  'batchSize', opts.batchSize, ...
  'gpus', opts.gpus,...
  'train', train_image_set, ...
  'val', val_image_set, ...
  'numEpochs', opts.numEpochs, ...
  'learningRate', opts.learningRate, ...
  'conserveMemory', opts.conserveMemory, ...
  'backPropDepth', opts.backPropDepth, ...
  'cudnn', opts.cudnn, ...
  'saveFreq', opts.saveFreq, ...
  'plotStatistics', opts.plotStatistics, ...
  'netParams', opts.netParams, ...
  'net_move', opts.net_move, ...
  'net_eval', opts.net_eval);

train_time = toc(start_time);
fprintf('\n-------------------------------------------------------\n')
fprintf('\n\n The training was completed in %.2f secs.\n\n',train_time);
fprintf('-------------------------------------------------------\n')
  
save(fullfile(opts.expDir,'net-final.mat'), 'net');


function [im, im_gt, aux] = getBatch(imdb,batch,Nbrs_idx)

N_gt = size(imdb.images.data,4);% Number of unique ground-truth images used 
% for training / testing.
im_gt = imdb.images.data(:,:,:,mod(batch-1,N_gt)+1);

% imdb.images.obs : The input of the first stage of the network
im = imdb.images.obs(:,:,:,batch); 
% Instead of using a vector noise_std of size N_gt*K (K : number of
% the different noise levels and N_gt the number of unique ground-truth 
% images) we use only a vector of size K. Then the first 1:N_gt images in 
% the data set are distorted by noise with standard deviation equal to 
% noise_std(1), the next N_gt+1:2*N_gt by noise with standard deviation 
% equal to im_noise_std(2), etc.
aux = struct('stdn',imdb.images.noise_std(ceil(batch/N_gt)), ...
             'Nbrs_idx',Nbrs_idx(:,:,:,batch));


% -------------------------------------------------------------------------
%                       Initialize Network
% -------------------------------------------------------------------------

function net = net_init_from_struct(opts)

net_add_layer = @unlnet_add_layer;
net.layers = {};
num_layers = numel(opts.net_struct);

for l = 1:num_layers
  
  switch opts.net_struct{l}.layer_type
    
    case 'unlnet'
      if ~isfield(opts.net_struct{l},'alpha')
        opts.net_struct{l}.alpha = opts.alpha;
      end       
      if ~isfield(opts.net_struct{l},'patchSize')
        opts.net_struct{l}.patchSize = opts.patchSize;
      end
      if ~isfield(opts.net_struct{l},'stride')
        opts.net_struct{l}.stride = opts.stride;
      end
      if ~isfield(opts.net_struct{l},'padSize')
        opts.net_struct{l}.padSize = opts.padSize;
      end
      if ~isfield(opts.net_struct{l},'padType')
        opts.net_struct{l}.padType = opts.padType;
      end      
      if ~isfield(opts.net_struct{l},'h')
        opts.net_struct{l}.h = opts.h;
      end
      if ~isfield(opts.net_struct{l},'ht')
        opts.net_struct{l}.ht = opts.ht;
      end      
      if ~isfield(opts.net_struct{l},'s')
        opts.net_struct{l}.s = opts.s;
      end
      if ~isfield(opts.net_struct{l},'st')
        opts.net_struct{l}.st = opts.st;
      end            
      if ~isfield(opts.net_struct{l},'weightSharing')
        opts.net_struct{l}.weightSharing = opts.weightSharing;
      end      
      if ~isfield(opts.net_struct{l},'g')
        opts.net_struct{l}.g = opts.h;
      end
      if ~isfield(opts.net_struct{l},'gt')
        opts.net_struct{l}.gt = opts.ht;
      end            
      if ~isfield(opts.net_struct{l},'groupWeightSharing')
        opts.net_struct{l}.groupWeightSharing = opts.groupWeightSharing;
      end
      if ~isfield(opts.net_struct{l},'weightNormalization')
        opts.net_struct{l}.weightNormalization = opts.weightNormalization;
      end      
      if ~isfield(opts.net_struct{l},'zeroMeanFilters')
        opts.net_struct{l}.zeroMeanFilters = opts.zeroMeanFilters;
      end            
      if ~isfield(opts.net_struct{l},'numFilters')
        opts.net_struct{l}.numFilters = opts.numFilters;
      end
      if ~isfield(opts.net_struct{l},'rbf_means')
        opts.net_struct{l}.rbf_means = opts.rbf_means;
      end
      if ~isfield(opts.net_struct{l},'rbf_precision')
        opts.net_struct{l}.rbf_precision = opts.rbf_precision;
      end
      if ~isfield(opts.net_struct{l},'rbf_weights')
        opts.net_struct{l}.rbf_weights = opts.rbf_weights;
      end
      if ~isfield(opts.net_struct{l},'learningRate')
        opts.net_struct{l}.learningRate = opts.lr;
      end
      if ~isfield(opts.net_struct{l},'first_stage')
        opts.net_struct{l}.first_stage = opts.first_stage;
      end
      if ~isfield(opts.net_struct{l},'shrink_type')
        opts.net_struct{l}.shrink_type = opts.shrink_type;
      end
      if ~isfield(opts.net_struct{l},'clb')
        opts.net_struct{l}.clb = opts.clb;
      end
      if ~isfield(opts.net_struct{l},'cub')
        opts.net_struct{l}.cub = opts.cub;
      end      
      if ~isfield(opts.net_struct{l},'Nbrs')
        opts.net_struct{l}.Nbrs = opts.Nbrs;
      end      
      
      net = net_add_layer(net, ...
        'alpha', opts.net_struct{l}.alpha, ...
        'layer_id', l, ...
        'inputSize', opts.inputSize, ...
        'layer_type', opts.net_struct{l}.layer_type, ...
        'cid', opts.cid, ...
        'patchSize',opts.net_struct{l}.patchSize, ...
        'numFilters', opts.net_struct{l}.numFilters, ...        
        'stride', opts.net_struct{l}.stride, ...
        'padSize', opts.net_struct{l}.padSize, ...
        'padType', opts.net_struct{l}.padType, ...
        'shrink_type', opts.net_struct{l}.shrink_type, ...
        'h', opts.net_struct{l}.h, ...
        's', opts.net_struct{l}.s, ...
        'ht', opts.net_struct{l}.ht, ...
        'st', opts.net_struct{l}.st, ...
        'g', opts.net_struct{l}.g, ...
        'gt', opts.net_struct{l}.gt, ...
        'zeroMeanFilters', opts.net_struct{l}.zeroMeanFilters, ...
        'weightNormalization', opts.net_struct{l}.weightNormalization, ...
        'weightSharing', opts.net_struct{l}.weightSharing, ...        
        'groupWeightSharing', opts.net_struct{l}.groupWeightSharing, ...                
        'rbf_means', opts.net_struct{l}.rbf_means, ...
        'rbf_precision', opts.net_struct{l}.rbf_precision, ...
        'rbf_weights', opts.net_struct{l}.rbf_weights, ...
        'learningRate', opts.net_struct{l}.learningRate, ...
        'first_stage', opts.net_struct{l}.first_stage, ...
        'clb', opts.net_struct{l}.clb, ... 
        'cub', opts.net_struct{l}.cub, ...
        'Nbrs', opts.net_struct{l}.Nbrs);

    case 'clip'
      if ~isfield(opts.net_struct{l},'lb')
        opts.net_struct{l}.lb = opts.lb;
      end
      if ~isfield(opts.net_struct{l},'ub')
        opts.net_struct{l}.ub = opts.ub;
      end
      net = net_add_layer(net,'layer_id',l, ...
        'layer_type', opts.net_struct{l}.layer_type, ...
        'lb',opts.net_struct{l}.lb, ...
        'ub',opts.net_struct{l}.ub);      
      
    case 'imloss'
      if ~isfield(opts.net_struct{l},'peakVal')
        opts.net_struct{l}.peakVal = opts.peakVal;
      end
      if ~isfield(opts.net_struct{l},'loss_type')
        opts.net_struct{l}.loss_type = opts.loss_type;
      end
      net = net_add_layer(net,'layer_id',l, ...
        'layer_type', opts.net_struct{l}.layer_type, ...
        'peakVal',opts.net_struct{l}.peakVal, ...
        'loss_type',opts.net_struct{l}.loss_type);
  end
end

% Meta parameters
net.meta.trainOpts.inputSize = opts.inputSize;
net.meta.trainOpts.noise_std = opts.noise_std;
net.meta.trainOpts.randSeed = opts.randn_seed;
net.meta.trainOpts.numEpochs = opts.numEpochs;
net.meta.trainOpts.learningRate = opts.learningRate;
net.meta.trainOpts.optimizer = char(opts.solver);
net.meta.netParams = opts.netParams;

function [knn_idx,knn_D] = precompute_knn_idx(im_net_input,opts)

[Nx,Ny,~,NI] = size(im_net_input);
Nx = Nx + sum(opts.padSize(1:2));
Ny = Ny + sum(opts.padSize(3:4));

patchDims= max(floor(([Nx,Ny]-opts.patchSize)./opts.stride)+1,0);

cid = class(im_net_input);

useGPU = ~isempty(opts.gpus);

if opts.cudnn, opts.cudnn = 'CuDNN'; else, opts.cudnn = 'NoCuDNN'; end

if useGPU
  gpuDevice(opts.gpus(1));
  knn_idx = gpuArray.zeros([patchDims,opts.Nbrs,NI],'uint32');
  if nargout > 1
    knn_D = cast(knn_idx,cid);
  end
  if ~isempty(opts.Wx)
    opts.Wx = cast(gpuArray(opts.Wx),cid);
  end
  if ~isempty(opts.Wy)
    opts.Wy = cast(gpuArray(opts.Wy),cid);
  end
else
  knn_idx = zeros([patchDims,opts.Nbrs,NI],'uint32');
  if nargout > 1
    knn_D = cast(knn_idx,cid);
  end
end

for t =1:opts.batchSize:NI
  batchStart = t;
  batchEnd = min(t+opts.batchSize-1, NI);
  ind = batchStart:batchEnd;
  if useGPU
    input = gpuArray(im_net_input(:,:,:,ind));
  else 
    input = (im_net_input(:,:,:,ind));
  end
  
  Nc = size(input,3);
  if Nc > 1 && ~opts.multichannel_dist
    input = sum(input,3)/Nc;
  end
  
  if nargout > 1
    [knn_idx(:,:,:,ind),knn_D(:,:,:,ind)] = misc.patchMatch(...
      nn_pad(input,opts.padSize,[],'padType',opts.padType),...
      'patchSize',opts.patchSize,'patchDist',opts.patchDist,...
      'Nbrs',opts.Nbrs,'searchwin',opts.searchwin,'stride',opts.stride,...
      'Wx',opts.Wx,'Wy',opts.Wy,'transform',opts.transform,'useSep',...
      opts.useSep,'sorted',opts.sorted,'cuDNN',opts.cudnn);
  else
    knn_idx(:,:,:,ind) = misc.patchMatch(nn_pad(input,opts.padSize,[],...
      'padType',opts.padType),'patchSize',opts.patchSize,'patchDist', ...
      opts.patchDist,'Nbrs',opts.Nbrs,'searchwin',opts.searchwin, ...
      'stride',opts.stride,'Wx',opts.Wx,'Wy',opts.Wy,'transform', ...
      opts.transform,'useSep',opts.useSep,'sorted',opts.sorted, ...
      'cuDNN',opts.cudnn);
  end
  
end
clear input;

if useGPU
  knn_idx = gather(knn_idx);
  if nargout > 1
    knn_D = gather(knn_D);
  end
end