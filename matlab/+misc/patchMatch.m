function [idx,dist] = patchMatch(x,varargin)
% Helper function for vl_nnconv_nlr and vl_nnconvt_nlr

%   `Stride`:: 1
%     The output stride or downsampling factor. If the value is a
%     scalar, then the same stride is applied to both vertical and
%     horizontal directions; otherwise, passing [STRIDEY STRIDEX]
%     allows specifying different downsampling factors for each
%     direction.
%
%   `patchSize`:: [8 8]
%
%   `Nbrs`:: 5
%     The number of closest neighbors used in the block-matching layer of
%     the non-local range convolution.
%
%   `searchwin`:: [15 15]
%     The half-dimensions of the search window in which the closest
%     neighbors are searched for.
%
%   `patchDist`:: {'euclidean'} | 'abs' | 'ncc'
%     It specifies the type of the distance that will be used for the image
%     patch similarity check.
%
%    `transform` ::  If set to true then the patch similarity takes place 
%     in the gradient domain instead of the image domain. (Default : false)
%
%     `cuDNN` :: 'cuDNN' || 'NocuDNN'. Flag which indicates if the layer
%      vl_nnblockMatch will use the cuDNN support or not. (Default: 'cuDNN')
%
%    `useSep` :: If set to true then the separable knnpatch is employed to 
%     find the closest neighbors.
%
%     `Wx,Wy` :: Weights used in the distance measure between two patches:
%                       Hp/2   Wp/2
%               d(a,b)= S      S  W(i,j)|f(a_x+i,a_y+j)-f(b_x+i,b_y+j)|^2
%                      i=-Hp/2 j=-Wp/2

opts.patchSize=[8,8];
opts.patchDist = 'euclidean';
opts.Nbrs=5; % Number of neighbors for the block-matching layer.
opts.searchwin=[15 15]; % Width and height of the search-window for similar
% patches.
opts.stride=[1,1];
opts.Wx = []; % kernel for weighting the patch elements.
opts.Wy = [];
%filterbank
opts.transform=false;
opts.useSep=true;
opts.sorted=true;
opts.cuDNN='cuDNN';
opts.excludeFromSearch = [0 0]; % If a different value than zero is provided 
%,i.e [kx,ky], then the search of similar patches for the patch with center
% i,j does not involve the patches who have centers less than
% i+kx,j+ky.

opts=vl_argparse(opts,varargin);

if opts.useSep
  if isempty(opts.Wx)
    opts.Wx=ones(opts.patchSize(1),1,'like',x);
  end
  if isempty(opts.Wy)
    opts.Wy=ones(1,opts.patchSize(2),1,'like',x);
  end
else
  if isempty(opts.Wx)
    opts.Wx=ones(opts.patchSize,'like',x);
  end
end
    

% The total number of patch comparisons is equal to 
% prod(ceil(2*(searchwin+1)./opts.stride)-1). However,
% for the corner pixels the valid number of patches is maxNumOfNeighbors.
maxNumOfNeighbors=prod(ceil(opts.searchwin./opts.stride));

if numel(opts.searchwin)==1
  opts.searchwin=opts.searchwin*ones(1,2);
end

if opts.Nbrs > maxNumOfNeighbors
  error('patchMatch:: The number of closest neighbors cannot be more than %d\n.', maxNumofNeighbors);
end

% Search for the opts.Nbrs most similar patches.
if opts.useSep
  if nargout > 1
    [idx,dist] = misc.knnpatch_sep(x,opts.patchSize,opts.searchwin,'stride',opts.stride,...
        'K',opts.Nbrs,'sorted',opts.sorted,'transform',opts.transform,...
        'patchDist',opts.patchDist,'cuDNN',opts.cuDNN,'Wx',opts.Wx,'Wy',...
        opts.Wy,'excludeFromSearch',opts.excludeFromSearch);
  else
    idx = misc.knnpatch_sep(x,opts.patchSize,opts.searchwin,'stride',opts.stride,...
        'K',opts.Nbrs,'sorted',opts.sorted,'transform',opts.transform,...
        'patchDist',opts.patchDist,'cuDNN',opts.cuDNN,'Wx',opts.Wx,'Wy',...
        opts.Wy,'excludeFromSearch',opts.excludeFromSearch);    
  end
else
  if nargout > 1
    [idx,dist] = misc.knnpatch(x,opts.patchSize,opts.searchwin,'stride',opts.stride,...
        'K',opts.Nbrs,'sorted',opts.sorted,'transform',opts.transform,...
        'patchDist',opts.patchDist,'cuDNN',opts.cuDNN,'W',opts.Wx,...
        'excludeFromSearch',opts.excludeFromSearch);
  else
    idx = misc.knnpatch(x,opts.patchSize,opts.searchwin,'stride',opts.stride,...
      'K',opts.Nbrs,'sorted',opts.sorted,'transform',opts.transform,...
      'patchDist',opts.patchDist,'cuDNN',opts.cuDNN,'W',opts.Wx,...
      'excludeFromSearch',opts.excludeFromSearch);
  end
end

% The size of y is going to be smaller than that of x since we keep only
% the valid part of the convolution. Therefore, we need to create a
% mapping between the pixels of interest in x to those in y. For example
% if stride=[sx,sy] and patchSize=[px py] then idx(1)=(px_,py_) where
% px_=floor((px+1)/2) and py_=floor((py+1)/2) is the center of the first
% valid patch in x. This however corresponds to the (1,1) pixel of the
% image y that results as the convolution of x with a filter of support
% [px, py] and stride=[sx, sy]. Accordingly, (px_+sx,py_) is the
% center of the second valid patch in x which corresponds to the (2,1)
% pixel of y. The correct mapping can be computed as follows :
%
%  i=rem(idx-1,size(x,1))+1 is the row in x and
%  j=floor((idx-1)/size(x,1))+1 is the column in x.
%
% Now the mapping i'=(i-px_)/sx + 1  and j'=(j-py_)/sy + 1 provides the
% correct pixel indices for y.

[Nx,Ny,~,NI]=size(x);
% Number of rows and columns of valid patches
patch_dims= max(floor(([Nx,Ny]-opts.patchSize)./opts.stride)+1,0);

Pc=floor((opts.patchSize+1)/2); % center of the patch
idx=uint32((rem(single(idx)-1,Nx)+1-Pc(1))/opts.stride(1)+1+...
  patch_dims(1)*((floor((single(idx)-1)/Nx)+1-Pc(2))/opts.stride(2)));

idx = reshape(idx,patch_dims(1),patch_dims(2),opts.Nbrs,NI);

if nargout > 1
  dist = reshape(dist,patch_dims(1),patch_dims(2),opts.Nbrs,NI);
end

