function [y,dh,dsh,dg] = nn_pgcf2Dt(x,h,sh,g,idx,dzdy,varargin)

% NN_PGCF2DT (Transpose of Group Collaborative Filtering)
%
%   Y = NN_PGCF2D(X, H, A, G, idx) extracts in total Np 2D-patches of size
%   PH x PW from the input X, groups the similar patches (specified by idx)
%   into 3D stacks (each stack has a size of PH x PW x Nbrs, where Nbrs is
%   the number of the closest neighbors to the reference patch) and
%   performs a 2D spatial transform (R: PH*PW -> M) to each patch of the
%   stack followed by a 1-D collaborative filtering across the group of
%   similar patches of the stack. The size of the final output Y is
%   H' x W' x M x N, where Np = H' x W' denotes the total number of
%   extracted patches.
%
%   Z = NN_PGCF2DT(Y, H, SH, G, SG, idx)
%
%   X is an array of dimension H x W x C x N where (H,W) are
%   the height and width of the image stack, C the number of the channels 
%   and N the number of images in the stack.
%
%   H is an array of dimension PH x PW x C x K where (PH,PW)
%   are the filter height and width and K the number of filters in the
%   bank.
%
%   SH is a vector of K elements used for the scaling of the filters H. If 
%   SH is set to the empty array then SH = ones(1,K).
%
%   G is an array of size Nbrs, where Nbrs is the number of the closest 
%   neighbors used in forming a 3D stack of similar patches. G is used 
%   during the collaborative filtering that takes place in the 3rd 
%   dimension of the patch group.
%
%   Idx is an array of size H'x W' x opts.Nbrs x N
%   where Np = H' x W' is the total number of patches. It is computed in 
%   the block-matching layer (misc.patchMatch).
%
%
%   If ConserveMemory is set to false (see below) then Y is a 3 x 1 cell 
%   array which keeps the outputs of some of the internal layers, necessary
%   for the backward mode.
%
%   [DZDX,DH,DSH,DG,DSG] = NN_NPGCFT(X, H, SH, G, SG, idx, DZDY)
%   computes the derivatives of the block projected onto DZDY. DZDX, DH, 
%   DSH, DG and DSG have the same dimensions as X, H, SH, G and SG 
%   respectively. In the backward mode X is 2x1 cell array where 
%   X{1}=Y{1} and X{2}=I where I is the input of the forward mode.
%
%   NN_PGCF2DT(..., 'option', value, ...) takes the following options:
%
%   `Stride`:: 4
%     The output stride or downsampling factor. If the value is a
%     scalar, then the same stride is applied to both vertical and
%     horizontal directions; otherwise, passing [STRIDEY STRIDEX]
%     allows specifying different downsampling factors for each
%     direction.
%
%   'derParams' :: [true,true,true]
%     If any entry of derParams is set to false then in the backward step
%     the corresponding parameter dh, ds or dg is not computed.
%
%   `PadSize`:: [0,0,0,0]
%     Specifies the amount of symmetric padding of the input X as
%     [TOP, BOTTOM, LEFT, RIGHT].
%
%   `PadType`:: 'symmetric'
%     Specifies the type of padding. Valid choices are 'zero'|'symmetric'.
%
%   `Nbrs`:: 8
%     The number of closest neighbors used in the block-matching layer of
%     the non-local range convolution.
%
%   `cuDNN`:: {'CuDNN'} | 'NoCuDNN' 
%     It indicates whether CuDNN will be used or not during the computation 
%     of the convolutions.
%
%   `WeightNormalization` :: [false,false]
%      If set to true then the filters F are normalized as F/||F||
%      and the weights G as G/Sum(G).
%
%   `zeroMeanFilters` :: false
%      If set to true then the mean of the filters H is subtracted before
%      applied to the input, i.e, H-E(H), where E is the mean operator.
%
%   `ConserveMemory`:: false | true
%       If set to true saves memory by not storing the intermediate results
%       from the different layers. If used in training then the value 
%       should be set to false.
%

% s.lefkimmiatis@skoltech.ru, 09/07/2016

% % Parameters that reproduce the steps of nn_pgcf2D.
%
% patchSize=[5,5];stride=[1,1];Nbrs=5;searchwin=[12 12];
% useGPU = false;
% h=misc.gen_dct2_kernel(patchSize,'classType',misc.getClass(x),'gpu',useGPU);
% h = h(:,:,:,2:end);
% sh = ones(size(h,4),'like',h);
% g = randn(Nbrs,1,'like',h);
% [idx, dist] = misc.patchMatch(x,'stride',stride,'Nbrs',Nbrs,'searchwin',searchwin,...
%   'patchsize',patchSize);
%
%  % Forward Mode
% y = nn_pgcf2D(x,h,sh,g,idx,[],'stride',stride,'Nbrs',Nbrs, ...
% 'cuDNN','cuDNN','weightNormalization',true(1,2),'zeroMeanFilters',true,...
% 'conserveMemory',false);
%
% % Backward Mode
% input = {y{1},x};
% dzdy = randn(size(y{end}),'like',y{end});
% [dzdx,dh,dg] = nn_pgcf2D(input,h,sh,g,idx,dzdy,'stride',stride,...
% 'Nbrs',Nbrs,'weightNormalization',true(1,2),'zeroMeanFilters',true);

opts.stride = [1,1];
opts.padSize = [0,0,0,0];
opts.padType = 'symmetric';
opts.Nbrs = 8; % Number of neighbors for the block-matching layer.
opts.cuDNN = 'cuDNN';
opts.derParams = true(1,3);
opts.conserveMemory = false;
opts.zeroMeanFilters = false;
opts.weightNormalization = false(1,2);

opts = vl_argparse(opts,varargin);

if numel(opts.derParams) < 4
  opts.derParams=[opts.derParams(:)',false(1,4-numel(opts.derParams))];
end

if numel(opts.weightNormalization) < 2
  opts.weightNormalization=[opts.weightNormalization(:)',...
    false(1,2-numel(opts.weightNormalization))];
end

nP = numel(opts.padSize);
if nP == 1
  opts.padSize = opts.padSize(1)*ones(1,4);
end

if numel(opts.stride) == 1
  opts.stride = opts.stride*[1,1];
end

if numel(g) ~= opts.Nbrs
  error(['nn_pgcf2Dt:: The number of the neighbor-filter coefficients ' ...
    'must be equal to the number of the closest neighbors.']);
end

gsum = sum(g(:));
if opts.weightNormalization(2)
  g = g/gsum;
end

if (isempty(dzdy) || nargin < 6)   
  dh = []; dg = []; dsh = []; 
  useGPU = isa(x,'gpuArray');
  % Number of rows and columns of valid patches
  patchDims = [size(x,1),size(x,2)];
  NI = size(x,4);
  
  if (size(idx,1) ~= patchDims(1) || size(idx,2) ~= patchDims(2) || ...
      size(idx,3) ~= opts.Nbrs || size(idx,4) ~= NI)
    error('nn_pcgf2Dt:: idx does not have the correct dimensions.');
  end
  
  [idx, n, I] = misc.FMapNLSumT_helper(idx);
  
  if opts.conserveMemory   
    
    if useGPU
      y = FMapNLSumT_gpu(x,g,idx-1,n,I-1);
    else
      y = FMapNLSumT_cpu(x,g,idx-1,n,I-1);
    end
    
    % Spatial transpose convolution - y is of size H x W x C x NI
    y = nn_conv2Dt(y,h,[],sh,[],'stride',opts.stride,'padSize', ...
      opts.padSize,'padType',opts.padType,'cuDNN',opts.cuDNN,...
      'zeroMeanFilters',opts.zeroMeanFilters,'weightNormalization',...
      opts.weightNormalization(1));
    
  else
    
    y = cell(2,1);
    
    if useGPU
      y{1} = FMapNLSumT_gpu(x,g,idx-1,n,I-1);
    else
      y{1} = FMapNLSumT_cpu(x,g,idx-1,n,I-1);
    end
    
    % Spatial transpose convolution - y{2} is of size H x W x C x NI
    y{2} = nn_conv2Dt(y{1},h,[],sh,[],'stride',opts.stride,'padSize', ...
      opts.padSize,'padType',opts.padType,'cuDNN',opts.cuDNN,...
      'zeroMeanFilters',opts.zeroMeanFilters,'weightNormalization',...
      opts.weightNormalization(1));
    
    % Re-arrange the entries of y to use them as inputs for the backward
    % step.
    
    %input = {y{1},x};    
  end    
else
  useGPU = isa(dzdy,'gpuArray');
  patchDims = size(x{1}(:,:,1,1));
  NI = size(dzdy,4);
    
  if (size(idx,1)~=patchDims(1) || size(idx,2)~=patchDims(2) || ...
      size(idx,3)~=opts.Nbrs || size(idx,4)~=NI)
    error('nn_npgcft:: idx does not have the correct dimensions.');
  end
  
  derParams = [opts.derParams(1) false opts.derParams(2)];
  [dzdy,dh,~,dsh] = nn_conv2Dt(x{1},h,[],sh,dzdy,'stride',opts.stride,...
    'padSize',opts.padSize,'padType',opts.padType,'cuDNN',opts.cuDNN,...
    'derParams',derParams,'zeroMeanFilters',opts.zeroMeanFilters,...
    'weightNormalization',opts.weightNormalization(1));
  x{1} = [];

  if opts.derParams(3)
    [widx,n,I] = misc.DerFMapNLSumT_helper(idx);
    dg = zeros(size(g),'like',g);

    if useGPU 
      DerFMapNLSumT = @DerFMapNLSumT_gpu; 
    else
      DerFMapNLSumT = @DerFMapNLSumT_cpu;
    end
    
    for k=1:opts.Nbrs
      tmp = DerFMapNLSumT(x{2},dzdy,widx(:,k,:)-1,n(:,k,:),...
        I(:,k,:)-1);
      dg(k) = sum(reshape(tmp,[],1));
    end
    
    if opts.weightNormalization(2)
      % dg = (I/sum(g) - ones(Nbrs,1)*g^T/sum(g)^2) * dg
      % Note that if weightNormalization = true then g = g/sum(g).
      dg = (dg - (sum(g(:).*dg(:))/gsum))/gsum;
    end
  else
    dg = [];
  end
  
  if useGPU
    y = FMapNLSum_gpu(dzdy,g,idx-1);
  else
    y = FMapNLSum_cpu(dzdy,g,idx-1);
  end
  
end
