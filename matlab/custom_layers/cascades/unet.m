function [y,dh1,dh2,ds1,ds2,dw,da,J,M] = unet(x,Obs,h1,h2,s1,s2,...
  rbf_weights,rbf_means,rbf_precision,stdn,alpha,dzdy,varargin)
%(Universal Denoising Network with Convolutional Layers) 
%
% Each UNET stage consists of the following layers:
% 1) NN_CONV2D 2) NN_CLIP 3) NN_SHRINK 4) NN_CONV2DT 5) DIFF 
% 6) NN_L2TRPROX
%   
%  If h2 is empty then the 1st and 4th layers share the same parameters h1
%
%  x    : input of the current stage
%  Obs  : input of the first stage of the network.
%  stdn : standard deviation of the noise distrorting the observed image
%  alpha: parameter for L2TRProx (Default value : 0)
%  h1,h2: Filters for nn_conv2D and nn_conv2Dt
%  s1,s2: Scaling coefficients for nn_conv2D and nn_conv2Dt.
%
%   Y = UNET(X,Obs,H1,H2,S1,S2,RBF_WEIGHTS,MEANS,PRECISION,STDN,ALPHA)
%
%   In the forward mode X is of size H x W x K x N (K: number of channels,
%   N: number of images). H, WH, are the inputs to the
%   layers nn_conv_frn, nn_convt_frn. RBF_weights
%   is a matrix of size B x M where M is the number of mixture components
%   in the RBF-mixture, RBF_means is of size M x 1 and rbf_precision is a
%   scalar. STDN is a vector N x 1 that is related to the standard 
%   deviation of the noise degrading each image.
%
%   If ConserveMemory is set to false (see below) then Y is a 4 x 1 cell
%   array which keeps the outputs of some of the internal layers, necessary
%   for the backward mode.
%
%   [DZDX,DH1,DH2,DS1,DS2,DW,DA] = UNET(X,Obs,H1,H2,S1,S2,RBF_WEIGHTS,MEANS, ...
%   PRECISION,STDN,ALPHA,DZDY) computes the derivatives of the stage
%   projected onto DZDY. DZDX, DH1, DH2, DS1, DS2, DW, DA and DZDY have the 
%   same dimensions as X, H1, H2, S1, S2, RBF_WEIGHTS, ALPHA and Y{4} 
%   respectively.
%   In the backward mode X is a cell array (4,1) where X{1} = Y{3},
%   X{2} = Y{2}, X{3} = Y{1}, X{4} = I,
%   where Y is the output of the forward mode with the conserveMemory 
%   option set to false and I is the input of the forward mode.
%
%   UNET(...,'OPT',VALUE,...) takes the following options:
%
%   `Jacobian`:: J (see RBFSHRINK)
%
%   `clipMask`:: M (Used for the backward pass of nn_clip). M is computed
%   during the forward pass of unet.
%
%   'Idx' :: Idx is an array of the same size as the output of the
%   NN_FiltResNorm layer and can be computed using the function
%   Idx=misc.gen_idx_4shrink(size(x));
%
%   `Stride`:: 1
%     The output stride or downsampling factor. If the value is a
%     scalar, then the same stride is applied to both vertical and
%     horizontal directions; otherwise, passing [STRIDEY STRIDEX]
%     allows specifying different downsampling factors for each
%     direction.
%
%   `Padsize:: Specifies the amount of padding of the input X as
%   [TOP, BOTTOM, LEFT, RIGHT].
%
%   `PadType`:: 'zero'
%     Specifies the type of padding. Valid choices are 'zero'|'symmetric'.
%
%   `WeightNormalization` :: false
%      If set to true then the filters are normalized as F/||F||.
%
%   `zeroMeanFilters` :: false
%      If set to true then the mean of the filters is subtracted before
%      applied to the input, i.e, F-E(F), where E is the mean operator.
%
%   'learningRate' :: 6x1 vector. if an element is set to zero then 
%   in the backward step the derivatives of the corresponding weights 
%   (dzdwh, dzdw, dzda) are not computed. (Default value : [1,1,1,1,1,1])
%
%   `ConserveMemory`:: false | true
%       If set to true saves memory by not storing the intermediate results
%       from the different layers. If used in training then the value
%       should be set to false.

% s.lefkimmiatis@skoltech.ru, 20/03/2017

% Example
% x = single(imread('/Users/stamatis/Documents/MATLAB/Data/BSDS500/gray/102061.jpg'));
% support = [5 5]; numFilters=24; padSize = [0,0,0,0]; stride=[1,1];
% zeroMeanFilters = true; weightNormalization = true;
% stdn = 25; y = x+noise_std*randn(size(x),'like',x);
% alpha = 0;
% cid = 'single';
% h = misc.gen_dct2_kernel(support,'classType',cid,'gpu',false);
% s = ones(1,numFilters,cid);
% rbf_means=cast(-310:10:310,cid); rbf_precision = 10;
% rbf_weights = randn(numFilters,numel(rbf_means),cid);
% obs = randn(size(x),'like',x);
% [y,~,~,~,~,~,~,J,M] = unet(x,obs,h,[],s,[],rbf_weights,rbf_means,...
%   rbf_precision,stdn,alpha,[],'stride',stride,'padSize',padSize,...
%   'padType',padType,'zeroMeanFilters',zeroMeanFilters,...
%   'weightNormalization',weightNormalization,'conserveMemory',false);
% dzdy = randn(size(y{end}),'like',x);
% input = {y{3},y{2},y{1},x};
% [y,dh,~,ds,~,dw,da] = unet(input,obs,h,[],s,[],rbf_weights,rbf_means,...
%   rbf_precision,stdn,alpha,[],'stride',stride,'padSize',padSize,...
%   'padType',padType,'zeroMeanFilters',zeroMeanFilters,...
%   'weightNormalization',weightNormalization);

opts.stride = [1,1];
opts.padSize = [0,0,0,0];
opts.padType = 'symmetric';
opts.cuDNN = 'cuDNN';
opts.Jacobian = [];
opts.clipMask = [];
opts.Idx=[];
opts.conserveMemory = false;
opts.first_stage = 0; % Denotes if this the first stage of the network.
opts.learningRate = [1,1,1,1,1,1];
opts.zeroMeanFilters = false;
opts.weightNormalization = false;
opts.data_mu = [];
opts.step = 0.2;
opts.origin = [];
opts.shrink_type = 'identity';
opts.lb = -inf;
opts.ub = inf;
%-----------------------------
opts = vl_argparse(opts,varargin);

if numel(opts.learningRate) ~= 6
  opts.learningRate = [opts.learningRate(:)' zeros(1,6-numel(opts.learningRate))];
end

switch opts.shrink_type
  case 'identity'
    Shrink = @nn_grbfShrink_lut;
  case 'residual'
    Shrink = @nn_grbfResShrink_lut;
  otherwise
    error('unet :: Unknown type of RBF shrinkage.');
end

assert(size(h1,4) == size(rbf_weights,1), ['Invalid input for ' ...
  'h1 - dimensions mismatch.']);
assert(isempty(h2) || all(size(h1) == size(h2)) , ['Invalid input for ' ...
  'h2 - dimensions mismatch.']);

weightSharing = isempty(h2);% If h2 is empty then conv2D and conv2Dt
% share the same weights.

if nargin < 12 || isempty(dzdy)
  dh1=[];dh2=[];ds1=[];ds2=[];dw=[];da=[];J=[];M=[];
  
  
  if opts.conserveMemory
    
    y = nn_conv2D(x,h1,[],s1,[],'stride',opts.stride,'padSize',...
      opts.padSize,'padType',opts.padType,'cuDNN',opts.cuDNN,...
      'zeroMeanFilters',opts.zeroMeanFilters,'weightNormalization',...
      opts.weightNormalization);
    
    y = nn_clip(y,opts.lb,opts.ub);    
    y = Shrink(y,rbf_weights,rbf_means,rbf_precision,[],...
      'Idx',opts.Idx,'data_mu',opts.data_mu,'step',opts.step,'origin',...
      opts.origin);
    
    if weightSharing
      y = nn_conv2Dt(y,h1,[],s1,[],'stride',opts.stride,'padSize',...
        opts.padSize,'padType',opts.padType,'cuDNN', opts.cuDNN,...
        'zeroMeanFilters',opts.zeroMeanFilters,'weightNormalization',...
        opts.weightNormalization);
    else
      y = nn_conv2Dt(y,h2,[],s2,[],'stride',opts.stride,'padSize',...
        opts.padSize,'padType',opts.padType,'cuDNN', opts.cuDNN,...
        'zeroMeanFilters',opts.zeroMeanFilters,'weightNormalization',...
        opts.weightNormalization);
    end
    
    y = nn_l2TrProx(x-y,Obs,stdn,alpha);
  
  else
    
    y = cell(4,1);
    
    y{1} = nn_conv2D(x,h1,[],s1,[],'stride',opts.stride,'padSize',...
      opts.padSize,'padType',opts.padType,'cuDNN',opts.cuDNN,...
      'zeroMeanFilters',opts.zeroMeanFilters,'weightNormalization',...
      opts.weightNormalization);
    
    if nargout > 8
      [y{1},M] = nn_clip(y{1},opts.lb,opts.ub);
    else
      y{1} = nn_clip(y{1},opts.lb,opts.ub);
    end
    
    if nargout > 7
      [y{2},~,J] = Shrink(y{1},rbf_weights,rbf_means,rbf_precision,[],...
        'Idx',opts.Idx,'data_mu',opts.data_mu,'step',opts.step,'origin',...
        opts.origin);
    else
      y{2} = Shrink(y{1},rbf_weights,rbf_means,rbf_precision,[],...
        'Idx',opts.Idx,'data_mu',opts.data_mu,'step',opts.step,'origin',...
        opts.origin);
    end
    
    if weightSharing
      y{3} = nn_conv2Dt(y{2},h1,[],s1,[],'stride',opts.stride,'padSize',...
        opts.padSize,'padType',opts.padType,'cuDNN', opts.cuDNN,...
        'zeroMeanFilters',opts.zeroMeanFilters,'weightNormalization',...
        opts.weightNormalization);
    else
      y{3} = nn_conv2Dt(y{2},h2,[],s2,[],'stride',opts.stride,'padSize',...
        opts.padSize,'padType',opts.padType,'cuDNN', opts.cuDNN,...
        'zeroMeanFilters',opts.zeroMeanFilters,'weightNormalization',...
        opts.weightNormalization);
    end
    
    % y = x^(t-1) - f(x^(t-1))
    y{3} = x-y{3};
    y{4} = nn_l2TrProx(y{3},Obs,stdn,alpha);
    % Re-arrange the entries of y to use them as inputs for the backward
    % step.
    
    %input = {y{3},y{2},y{1},x};   
  end
else
  J=[];M=[];

  [dzdy,da] = nn_l2TrProx(x{1},Obs,stdn,alpha,dzdy,'derParams',...
    logical(opts.learningRate(6)));
  x{1} = [];
  
  y = -dzdy;
  
  if opts.first_stage
    clear dzdy;
  end

  if weightSharing
    lr = [opts.learningRate(1),false,opts.learningRate(3)];
    [y,dh2,~,ds2] = nn_conv2Dt(x{2},h1,[],s1,y,'stride',opts.stride,...
      'padSize',opts.padSize,'padType',opts.padType,'cuDNN',opts.cuDNN,...
      'derParams',logical(lr) ,'zeroMeanFilters',...
      opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization);
  else
    lr = [opts.learningRate(2),false,opts.learningRate(4)];
    [y,dh2,~,ds2] = nn_conv2Dt(x{2},h2,[],s2,y,'stride',opts.stride,...
      'padSize',opts.padSize,'padType',opts.padType,'cuDNN',opts.cuDNN,...
      'derParams',logical(lr),'zeroMeanFilters',...
      opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization);
  end
  x{2} = [];

  [y, dw] = Shrink(x{3},rbf_weights,rbf_means,rbf_precision,y,...
    'Jacobian',opts.Jacobian,'Idx',opts.Idx,'derParams',...
    logical(opts.learningRate(5)),'data_mu',opts.data_mu,'step',...
    opts.step,'origin',opts.origin);
  x{3} = [];
  
  y = nn_clip([],opts.lb,opts.ub,y,'mask',opts.clipMask); 
    
  lr = [opts.learningRate(1),false,opts.learningRate(3)];
  [y,dh1,~,ds1] = nn_conv2D(x{4},h1,[],s1,y,'stride',opts.stride,...
    'padSize',opts.padSize,'padType',opts.padType,'cuDNN',opts.cuDNN,...
    'derParams',logical(lr),'zeroMeanFilters', ...
    opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization);
  
  clear x;
  
  if weightSharing
    dh1 = dh1 + dh2;
    dh2 = [];
    ds1 = ds1 + ds2;
    ds2 = [];
  end
  
  % If this is the first stage of the network then we don't need to
  % correctly compute dzdx and therefore we save computations.
  if opts.first_stage
    y = [];
  else
    y = y+dzdy;
  end
  
end

