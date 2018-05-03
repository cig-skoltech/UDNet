function [y,dzdw,J,M] = unlnet(x,Obs,weights,rbf_means,rbf_precision, ...
  stdn,dzdy,varargin)

% weights={h1,s1,g1,h2,s2,g2,rbfW,alpha};
% dzdw = {dh1,ds1,dg1,dh2,ds2,dg2,dw,da}
%(Universal Non-Local Denoising) 
%
% Each unlnet stage consists of the following layers:
% 1) NN_PGCF2D 2) NN_CLIP 3) NN_SHRINK_LUT 4) NN_PGCF2DT 5) DIFF
% 6) NN_L2TRPROX
%
%  If any of h2 and g2 are empty then the 1st and 4th layers share the  
%  same parameter h1 or/and g1.
%
%  x      : input of the current stage
%  Obs    : input of the first stage of the network.
%  stdn   : standard deviation of the noise degrading the network input.
%  alpha  : parameter for trainable layer nn_l2trpox 
%  h1,h2  : Filters for nn_conv2D and nn_conv2Dt
%  s1,s2  : Scaling coefficients for nn_conv2D and nn_conv2Dt.
%  g1,g2  : Filters for FMapNLSum and FMapNLSumT.

%
%   Y = UNLNET(X,Obs,H1,S1,G1,H2,S2,G2,RBFW,MEANS,PRECISION,ALPHA,STDN)
%
%   In the forward mode X is of size H x W x C x N (C: number of channels,
%   N: number of images). H1, H2, S1, S2, G1, G2, K1, K2 are the inputs to 
%   the layers nn_pgcf2D, nn_pgcf2Dt. RBFW
%   is a matrix of size B x M where M is the number of mixture components
%   in the RBF-mixture, RBF_means is of size M x 1 and rbf_precision is a
%   scalar. Alpha is a scalar that is a learnable parameter of the 
%   NN_l2TrPox layer.
%
%   If ConserveMemory is set to false (see below) then Y is a 4 x 1 cell
%   array which keeps the outputs of some of the internal layers, necessary
%   for the backward mode.
%
%   [DZDX, DH1, DS1, DG1, DH2 DS2, DG2, DW, DA] = UNLNET(X,Obs, ...
%   H1,S1,G1,H2,S2,G2,RBFW,MEANS,PRECISION,ALPHA,STDN,DZDY) 
%   computes the derivatives of the stage projected onto DZDY. DZDX, 
%   DH1, DH2, DS1, DS2, DG1, DG2, DW, DA and DZDY have the
%   same dimensions as X, H1, H2, S1, S2, G1, G2, RBFW, 
%   ALPHA and Y{4} respectively. In the backward mode X is a cell array 
%   (4,1) where X{1} = Y{3}, X{2} = Y{2}, X{3} = Y{1}, X{4} = I,
%   where Y is the output of the forward mode with the conserveMemory 
%   option set to false and I is the input of the forward mode. 
%
%   UNLNET(...,'OPT',VALUE,...) takes the following options:
%
%   `Jacobian`:: J (see RBFSHRINK)
%
%   'Idx' :: Idx is an array of the same size as the output of the
%       NN_FiltResNorm layer and can be computed using the function
%       Idx=misc.gen_idx_4shrink(size(x));
%
%   `Nbrs_idx` :: It is the output of misc.patchMatch applied on
%       vl_nnpad(x,opts.padSize);
%
%   `Stride`:: 1
%       The output stride or downsampling factor. If the value is a
%       scalar, then the same stride is applied to both vertical and
%       horizontal directions; otherwise, passing [STRIDEY STRIDEX]
%       allows specifying different downsampling factors for each
%       direction.
%
%   `Nbrs`:: 8
%       The number of closest neighbors used in the block-matching layer of
%       the non-local range convolution.
%
%   `searchwin`:: [15 15]
%       The half-dimensions of the search window in which the closest
%       neighbors are searched for.
%
%   `patchDist`:: {'euclidean'} | 'abs'
%       It specifies the type of the distance that will be used for the image
%       patch similarity check.
%
%   `Padsize:: Specifies the amount of padding of the input X as
%       [TOP, BOTTOM, LEFT, RIGHT].
%
%   `PadType`:: 'symmetric'
%     Specifies the type of padding. Valid choices are 'zero'|'symmetric'.
%
%   `WeightNormalization` :: [false,false]
%      If set to true then the filters are normalized as F/||F|| and G/SUM(G).
%
%   `zeroMeanFilters` :: false
%      If set to true then the mean of the filters is subtracted before
%      applied to the input, i.e, F-E(F), where E is the mean operator.
%
%   `transform` ::  If set to true then the patch similarity takes place
%       in the gradient domain instead of the image domain. (Default : false)
%
%   `useSep` :: If set to true then the separable knnpatch is employed to
%       find the closest neighbors.
%
%   `sorted` :: If set to true then the neighbors are sorted according to
%       their distance from the patch of interest.
%
%   'learningRate' :: 10x1 vector. if an element is set to zero then 
%       in the backward step the derivatives of the corresponding weights 
%       (dh1, dh2, ds1, ds2, dg1, dg2, dk1, dk2, dw, da) are not computed. 
%       (Default value : [1,1,1,1,1,1,1,1,1,1])
%
%   `ConserveMemory`:: false | true
%       If set to true saves memory by not storing the intermediate results
%       from the different layers. If used in training then the value
%       should be set to false.

% s.lefkimmiatis@skoltech.ru, 30/06/2017

opts.stride = [1,1];
opts.padSize = [0,0,0,0];
opts.padType = 'symmetric';
opts.cuDNN = 'cuDNN';
opts.Jacobian = [];
opts.clipMask = [];
opts.Idx=[];
opts.conserveMemory = false;
opts.first_stage = false; % Denotes if this is the first stage of the network.
opts.learningRate = ones(1,8);
opts.zeroMeanFilters = false;
opts.weightNormalization = false(1,2);
opts.data_mu = [];
opts.step = [];
opts.origin = [];
opts.shrink_type = 'identity';
opts.lb = -inf;
opts.ub = inf;

% Params for patchMatch
opts.searchwin = [15,15];
opts.patchDist = 'euclidean';
opts.transform = false;
opts.useSep = true;
opts.sorted = true;
opts.Wx = []; % kernel for weighting the patch elements.
opts.Wy = [];
opts.Nbrs_idx = [];
opts.Nbrs = [];
%-----------------------------
opts = vl_argparse(opts,varargin);

if numel(opts.learningRate) ~= 8
  opts.learningRate = [opts.learningRate(:)' zeros(1,8-numel(opts.learningRate))];
end

switch opts.shrink_type
  case 'identity'
    Shrink = @nn_grbfShrink_lut;
  case 'residual'
    Shrink = @nn_grbfResShrink_lut;
  otherwise
    error('unlnet :: Unknown type of RBF shrinkage.');
end

if isempty(opts.Nbrs) 
  if isempty(opts.Nbrs_idx)
    error('unlnet :: The number of closest neighbors must be specified.');
  else
    opts.Nbrs = size(opts.Nbrs_idx,3);
  end
end


h1 = weights{1}; s1 = weights{2}; g1 = weights{3};
h2 = weights{4}; s2 = weights{5}; g2 = weights{6};
rbfW = weights{7}; alpha = weights{8};
clear weights;

assert(size(h1,4) == size(rbfW,1), ['Invalid input for ' ...
  'h1 - dimensions mismatch.']);
assert( isempty(h2) || all(size(h2) == size(h1)), ...
  'Invalid input for h2 - dimensions mismatch.');

weightSharing = isempty(h2);% If h2 is empty then pgcf2D and pgcf2Dt
% share the same spatial weights.
GroupWeightSharing = isempty(g2);% If g2 is empty then pgcf2D and pgcf2Dt
% share the same group weights.

assert( numel(g1) == opts.Nbrs, 'Invalid input for g1 - dimensions mismatch.')

assert( isempty(g2) || all(size(g2) == size(g1)), ...
  'Invalid input for g2 - dimensions mismatch.')


if nargin < 6 || isempty(dzdy)
  dzdw = []; J = []; M = [];
   
  % Block-matching
  if isempty(opts.Nbrs_idx)    
    opts.Nbrs_idx = misc.patchMatch(nn_pad(x,opts.padSize,[],'padType',...
      opts.padType),'patchSize', [size(h1,1) size(h1,2)],'searchwin', ...
      opts.searchwin,'stride', opts.stride,'Nbrs',opts.Nbrs,'patchDist', ...
      opts.patchDist,'transform',opts.transform,'cuDNN',opts.cuDNN, ...
      'useSep',opts.useSep,'sorted', opts.sorted,'Wx',opts.Wx,'Wy',opts.Wy);    
  end  
    
  if opts.conserveMemory
    y = nn_pgcf2D(x,h1,s1,g1,opts.Nbrs_idx,[],'stride', opts.stride, ...
      'padSize',opts.padSize,'padType',opts.padType,'Nbrs',opts.Nbrs, ...
      'cuDNN',opts.cuDNN,'conserveMemory',true,'zeroMeanFilters', ...
      opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization);
    
    if nargout > 3
      [y,M] = nn_clip(y,opts.lb,opts.ub);
    else
      y = nn_clip(y,opts.lb,opts.ub);
    end
    
    if nargout > 2
      [y,~,J] = Shrink(y,rbfW,rbf_means,rbf_precision,[],...
        'Idx',opts.Idx,'data_mu',opts.data_mu,'step',opts.step,'origin',...
        opts.origin);
    else
      y = Shrink(y,rbfW,rbf_means,rbf_precision,[],...
        'Idx',opts.Idx,'data_mu',opts.data_mu,'step',opts.step,'origin',...
        opts.origin);
    end
    
    if weightSharing && GroupWeightSharing
      y = nn_pgcf2Dt(y,h1,s1,g1,opts.Nbrs_idx,[],'stride',opts.stride,...
        'padSize',opts.padSize,'padType',opts.padType,'Nbrs',opts.Nbrs, ...
        'cuDNN',opts.cuDNN,'conserveMemory',true,'zeroMeanFilters', ...
        opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization);
    elseif weightSharing && ~GroupWeightSharing
      y = nn_pgcf2Dt(y,h1,s1,g2,opts.Nbrs_idx,[],'stride',opts.stride,...
        'padSize',opts.padSize,'padType',opts.padType,'Nbrs',opts.Nbrs, ...
        'cuDNN',opts.cuDNN,'conserveMemory',true,'zeroMeanFilters', ...
        opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization);
    elseif ~weightSharing && GroupWeightSharing
      y = nn_pgcf2Dt(y,h2,s2,g1,opts.Nbrs_idx,[],'stride',opts.stride,...
        'padSize',opts.padSize,'padType',opts.padType,'Nbrs',opts.Nbrs, ...
        'cuDNN',opts.cuDNN,'conserveMemory',true,'zeroMeanFilters', ...
        opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization);
    else
      y = nn_pgcf2Dt(y,h2,s2,g2,opts.Nbrs_idx,[],'stride',opts.stride,...
        'padSize',opts.padSize,'padType',opts.padType,'Nbrs',opts.Nbrs, ...
        'cuDNN',opts.cuDNN,'conserveMemory',true,'zeroMeanFilters', ...
        opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization);
    end
    
    y = nn_l2TrProx(x-y,Obs,stdn,alpha);   
    
  else
    y=cell(4,1);
    
    y{1} = nn_pgcf2D(x,h1,s1,g1,opts.Nbrs_idx,[],'stride', opts.stride, ...
      'padSize',opts.padSize,'padType',opts.padType,'Nbrs',opts.Nbrs, ...
      'cuDNN',opts.cuDNN,'conserveMemory',false,'zeroMeanFilters', ...
      opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization);
    
   if nargout > 3
      [y{1}{2},M] = nn_clip(y{1}{2},opts.lb,opts.ub);
    else
      y{1}{2} = nn_clip(y{1}{2},opts.lb,opts.ub);
    end 
        
    if nargout > 2
      [y{2},~,J] = Shrink(y{1}{2},rbfW,rbf_means,...
        rbf_precision,[],'Idx',opts.Idx,'data_mu',opts.data_mu,'step',...
        opts.step,'origin',opts.origin);
    else
      y{2} = Shrink(y{1}{2},rbfW,rbf_means,...
        rbf_precision,[],'Idx',opts.Idx,'data_mu',opts.data_mu,'step',...
        opts.step,'origin',opts.origin);
    end
    
    if weightSharing && GroupWeightSharing
      y{3} = nn_pgcf2Dt(y{2},h1,s1,g1,opts.Nbrs_idx,[],'stride',opts.stride,...
        'padSize',opts.padSize,'padType',opts.padType,'Nbrs',opts.Nbrs, ...
        'cuDNN',opts.cuDNN,'conserveMemory',false,'zeroMeanFilters', ...
        opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization);
    elseif weightSharing && ~GroupWeightSharing
      y{3} = nn_pgcf2Dt(y{2},h1,s1,g2,opts.Nbrs_idx,[],'stride',opts.stride,...
        'padSize',opts.padSize,'padType',opts.padType,'Nbrs',opts.Nbrs, ...
        'cuDNN',opts.cuDNN,'conserveMemory',false,'zeroMeanFilters', ...
        opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization);
    elseif ~weightSharing && GroupWeightSharing
      y{3} = nn_pgcf2Dt(y{2},h2,s2,g1,opts.Nbrs_idx,[],'stride',opts.stride,...
        'padSize',opts.padSize,'padType',opts.padType,'Nbrs',opts.Nbrs, ...
        'cuDNN',opts.cuDNN,'conserveMemory',false,'zeroMeanFilters', ...
        opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization);
    else
      y{3} = nn_pgcf2Dt(y{2},h2,s2,g2,opts.Nbrs_idx,[],'stride',opts.stride,...
        'padSize',opts.padSize,'padType',opts.padType,'Nbrs',opts.Nbrs, ...
        'cuDNN',opts.cuDNN,'conserveMemory',false,'zeroMeanFilters', ...
        opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization);
    end
    
    % y = x^(t-1) - f(x^(t-1))
    y{3}{2} = x-y{3}{2};
    y{4} = nn_l2TrProx(y{3}{2},Obs,stdn,alpha);
    % Re-arrange the entries of y to use them as inputs for the backward
    % step.
    
    %input = {y{3}{2},{y{3}{1},y{2}},y{1}{2},{y{1}{1},x}};   
  end
  
else
  J = []; M = [];
  
  [dzdy,dzdw{8}] = nn_l2TrProx(x{1},Obs,stdn,alpha,dzdy,'derParams',...
    logical(opts.learningRate(8)));
  x{1} = [];
  
  y = -dzdy;
  
  if opts.first_stage
    clear dzdy;
  end

  if weightSharing && GroupWeightSharing
    [y,dzdw{4},dzdw{5},dzdw{6}] = nn_pgcf2Dt(x{2},h1,s1,g1,opts.Nbrs_idx,y, ...
      'stride',opts.stride,'padSize',opts.padSize,'padType', ...
      opts.padType,'Nbrs',opts.Nbrs,'cuDNN',opts.cuDNN,'zeroMeanFilters', ...
      opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization, ...
      'derParams',logical(opts.learningRate([1,2,3])));
  elseif weightSharing && ~GroupWeightSharing
    [y,dzdw{4},dzdw{5},dzdw{6}] = nn_pgcf2Dt(x{2},h1,s1,g2,opts.Nbrs_idx,y, ...
      'stride',opts.stride,'padSize',opts.padSize,'padType', ...
      opts.padType,'Nbrs',opts.Nbrs,'cuDNN',opts.cuDNN,'zeroMeanFilters', ...
      opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization, ...
      'derParams',logical(opts.learningRate([1,2,6])));
  elseif ~weightSharing && GroupWeightSharing
    [y,dzdw{4},dzdw{5},dzdw{6}] = nn_pgcf2Dt(x{2},h2,s2,g1,opts.Nbrs_idx,y, ...
      'stride',opts.stride,'padSize',opts.padSize,'padType', ...
      opts.padType,'Nbrs',opts.Nbrs,'cuDNN',opts.cuDNN,'zeroMeanFilters', ...
      opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization, ...
      'derParams',logical(opts.learningRate([4,5,3])));
  else
    [y,dzdw{4},dzdw{5},dzdw{6}] = nn_pgcf2Dt(x{2},h2,s2,g2,opts.Nbrs_idx,y, ...
      'stride',opts.stride,'padSize',opts.padSize,'padType', ...
      opts.padType,'Nbrs',opts.Nbrs,'cuDNN',opts.cuDNN,'zeroMeanFilters', ...
      opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization, ...
      'derParams',logical(opts.learningRate([4,5,6])));
  end
  x{2} = [];
    
  [y, dzdw{7}] = Shrink(x{3},rbfW,rbf_means,rbf_precision,y,...
    'Jacobian',opts.Jacobian,'Idx',opts.Idx,'derParams',...
    logical(opts.learningRate(7)),'data_mu',opts.data_mu,'step',...
    opts.step,'origin',opts.origin);
  x{3} = [];
  
  y = nn_clip([],opts.lb,opts.ub,y,'mask',opts.clipMask); 
  
  [y,dzdw{1},dzdw{2},dzdw{3}] = nn_pgcf2D(x{4},h1,s1,g1,opts.Nbrs_idx,y, ...
    'stride',opts.stride,'padSize',opts.padSize,'padType', ...
    opts.padType,'Nbrs',opts.Nbrs,'cuDNN',opts.cuDNN,'zeroMeanFilters', ...
    opts.zeroMeanFilters,'weightNormalization',opts.weightNormalization, ...
    'derParams',logical(opts.learningRate([1,2,3])));
  
  clear x;
  
  if weightSharing && GroupWeightSharing
    dzdw{1} = dzdw{1} + dzdw{4};
    dzdw{4} = [];
    dzdw{2} = dzdw{2} + dzdw{5};
    dzdw{5} = [];
    dzdw{3} = dzdw{3} + dzdw{6};
    dzdw{6} = [];
  elseif weightSharing && ~GroupWeightSharing
    dzdw{1} = dzdw{1} + dzdw{4};
    dzdw{4} = [];
    dzdw{2} = dzdw{2} + dzdw{5};
    dzdw{5} = [];
  elseif ~weightSharing && GroupWeightSharing
    dzdw{3} = dzdw{3} + dzdw{6};
    dzdw{6} = [];
  end
  
  % If this is the first stage of the network then we don't need to
  % correctly compute dzdx and therefore we save computations.
  if opts.first_stage
    y = [];
  else
    y = y+dzdy;
  end
  
end

