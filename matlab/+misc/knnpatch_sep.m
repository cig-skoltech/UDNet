function [knn,knn_D,LUT]=knnpatch_sep(f,patchSize,searchwin,varargin)
% [KNN,KNN_D]=KNNPATCH(f,patchSize,searchwin) returns a KNN-field 
%(K-Nearest Neighbors) for each patch of size Hp x Wp in a search window of
%size (2*Hw+1) x (2*Ww+1).
%
% ========================== INPUT PARAMETERS (required) ==================
% Parameters    Values description
% =========================================================================
% f             Vector/Scalar-valued image of size Nx x Ny x Nc x B, 
%               where Nc: number of channels and B: number of images.
% patchSize     Vector [Hp Wp] where Hp and Wp is the height and width of  
%               the patches extracted from the image f.
% searchwin     Vector [Hw Ww] where 2*Hw+1 and 2*Ww+1 is the height and 
%               width of the search window.
% ======================== OPTIONAL INPUT PARAMETERS ======================
% Parameters    Values' description
% stride        [Sx Sy] where Sx and Sy are the step-sizes in x- and
%               y-directions. (Default: stride=[1 1]).
% padSize       specifies the amount of padding of the image as 
%               [TOP, BOTTOM, LEFT, RIGHT]. 
% padType       'zero' || 'symmetric' indicates the type of padding.
%               (Default: padType='zero').
% W             Weights used in the distance measure between two patches:
%                       Hp/2   Wp/2
%               d(a,b)= S      S  W(i,j)|f(a_x+i,a_y+j)-f(b_x+i,b_y+j)|^2
%                      i=-Hp/2 j=-Wp/2
%               (Default: nweights=ones(patchSize)). Note that W
%               must be a symmetric matrix. 
% K             Number of closest neighbors (Default: 10). 
% sorted        If sorted is set to true then knn is sorted based on the
%               patch-distance (from smaller to larger).
% patchDist    'euclidean' || 'abs' indicates the type of the patch 
%               distance (Default:'euclidean').
% transform     If set to true then the patch similarity takes place in the
%               gradient domain. (Default : false)
% cuDNN         'cuDNN' || 'NocuDNN'. Flag which indicates if the function 
%               will use the cuDNN support or not. (Default: 'cuDNN')
% =========================================================================
% ========================== OUTPUT PARAMETERS ============================
% knn           KNN field of size Np x K x B which contains the 
%               coordinates of the center of the K closest patches
%               (including the patch itself). 
% knn_D         KNN field of size Np x K x B which contains the distances
%               of the K closest patches (including the patch itself).
% LUT           Look-up table which maps the image coordinates of the patch
%               centers to the index of the patches extracted by im2patch.
% =========================================================================
%
% stamatis@math.ucla.edu, 30/12/2015
%
% =========================================================================

% We can make a look-up table which maps the coordinates of the patch 
% centers to the index of the patches extracted by im2patch. This way we 
% can directly map the patch coordinates stored in knn(:,:,1) to the 
% specific patch number stored in the array P=im2patch_cpu(f,patchSize,stride);
%
% f=double(imread('peppers.png'));f=f/max(f(:));
% patchSize=[8 8];stride=[3,2]; pad=[0, 0, 0, 0];searchwin=[10 10];K=10;
% [knn,knn_D,LUT]=misc.knnpatch(f,patchSize,searchwin,'stride',stride,'K',K);
% 
% %% ***  Look Up Table ***
% LUT=zeros(size(f,1)*size(f,2),1);
% T=knn(:,1,1);
% for k=1:size(T), LUT(T(k))=k; end
% %% ***  ***  ***
% 
% P=im2patch_gpu(f,patchSize,stride,pad);
% idx1=randsrc(1,1,1:size(knn,1)); % choose randomly a patch.
% idx2=randsrc(1,1,1:K); % choose randomly a neighbor of this patch.
% sub=knn(idx1,idx2);
% % Extract the patch from the image 
% i=rem(double(sub)-1,size(f,1))+1;
% j=floor((double(sub)-1)/size(f,1))+1;
% fpatch=f(i-floor((patchSize(1)-1)/2):i+ceil((patchSize(1)-1)/2),j-floor((patchSize(2)-1)/2):j+ceil((patchSize(2)-1)/2),:);
% % Verify the correctness of the mapping
% reshape(P(LUT(sub),:,:),[patchSize size(f,3)])-fpatch
% % Verify correctness of the similarity metric 
% dist=@(P1,P2) sum((P1(:)-P2(:)).^2)
% e=knn_D(idx1,idx2)-dist(P(LUT(knn(idx1,1)),:,:),P(LUT(knn(idx1,idx2)),:,:));
% e % this should be close to zero.


% Plot a group of floor((size(h,1)-1)/2), ceil((size(h,1)-1)/2)f similar patches
% f=double(imread('peppers.png'));f=f/max(f(:));
% patchSize=[8 8];stride=[1,1]; searchwin=[15 15];K=16;
% [knn,LUT]=knnpatch(f,patchSize,searchwin,'stride',stride,'K',K);
% P=im2patch(f,patchSize,stride);
% idx1=randsrc(1,1,1:size(knn,1)); % choose randomly a patch.
% V=P(LUT(knn(idx1,:,1)),:,:,);
% V=reshape(V,[patchSize size(f,3) K]);
% J=vl_imarray(V);
% imshow(J/255,[]);
% % or 
% D1=cat(1,V(:,:,:,1),V(:,:,:,2),V(:,:,:,3),V(:,:,:,4));
% D2=cat(1,V(:,:,:,5),V(:,:,:,6),V(:,:,:,7),V(:,:,:,8));
% D3=cat(1,V(:,:,:,9),V(:,:,:,10),V(:,:,:,11),V(:,:,:,12));
% D4=cat(1,V(:,:,:,13),V(:,:,:,14),V(:,:,:,15),V(:,:,:,16));
% PG=cat(2,D1,D2,D3,D4);
% figure, imshow(PG,[]);

% assert(patchSize(1)==patchSize(2),['knnpatch:: InvalidInput. ', ...
%   'The patch dimensions must be equal ']);
  
opts.stride=[1 1];
opts.Wx=ones(patchSize(1),1,'like',f);
opts.Wy=ones(1,patchSize(2),1,'like',f);
opts.K=10;
opts.padSize= [0,0,0,0];
opts.padType = 'zero';
opts.sorted=false;
opts.patchDist = 'euclidean';
opts.transform = false;
opts.cuDNN='cuDNN';
opts.excludeFromSearch = [0 0]; % If a different value than zero is provided 
%,i.e [kx,ky], then the search of similar patches for the patch with center
% i,j does not involve the patches who have centers whose coordinates
% [px,py] satisfy the condition px <= i+kx && py <= j+ky.

opts=vl_argparse(opts,varargin);

if numel(opts.Wx)==1
  opts.Wx=opts.Wx*ones(patchSize(1),1,'like',f);
end

if numel(opts.Wy)==1
  opts.Wy=opts.Wy*ones(patchSize(1),1,'like',f);
end

if numel(opts.Wx)~=patchSize(1)
  error('knnpatch:InvalidInput','Wx must be a vector of size equal to patchSize(1).');
end

if numel(opts.Wy)~=patchSize(2)
  error('knnpatch:InvalidInput','Wy must be a vector of size equal to patchSize(1).');
end
  
if numel(opts.padSize)==1
  opts.padSize=opts.padSize*ones(1,4);
end

if (opts.padSize(1) > size(f,1) || opts.padSize(2) > size(f,1) ...
    || opts.padSize(3) > size(f,2) || opts.padSize(4) > size(f,2))
  error('knnpatch:InvalidInput','padSize cannot be greater than inputSize.');
end

sflag=false; % Check for equal-size padding on TOP-BOTTOM and LEFT-RIGHT
if opts.padSize(1)==opts.padSize(2) && opts.padSize(3)==opts.padSize(4)
  sflag=true;
end

if sum(opts.padSize)~=0 % If sum(opts.padsize)==0 then there is no padding.
  % Pad the image according to padSize
  if sflag
    f=padarray(f,[opts.padSize(1),opts.padSize(3)],opts.padType,'both');
  else
    f = padarray(f,[opts.padSize(1),opts.padSize(3)],opts.padType,'pre');
    f = padarray(f,[opts.padSize(2),opts.padSize(4)],opts.padType,'post');
  end
end

[Nx, Ny, Nc, NI] = size(f);

patch_dims= max(floor(([Nx Ny]-patchSize)./opts.stride)+1,0);
patch_num=prod(patch_dims);

opts.K=opts.K-1;
if isa(f,'gpuArray')
    knn=gpuArray.zeros(patch_num,opts.K,NI,'uint32');
else
    knn=zeros(patch_num,opts.K,NI,'uint32');
end

knn_D=zeros(patch_num,opts.K,NI,'like',f);

w=patchSize;
wc=floor((w+1)/2);% the center of the weighting function.

% vector of size equal to patch_num in which we store the image coordinates
% of the centers of the valid image patches.
T=uint32(1:Nx*Ny);
T=reshape(T,[Nx Ny]);
T=T(wc(1):opts.stride(1):end-(w(1)-wc(1)),wc(2):opts.stride(2):end-(w(2)-wc(2)));
T=T(:);

% If K = 1 then we don't need to make any search.
if opts.K == 0
  if isa(f,'gpuArray')
    T = gpuArray(T);
  end
  knn = repmat(T,[1 1 NI]);
  LUT = [];
  return; 
end

tmp = 0:opts.stride(1):searchwin(1);
tmp = tmp(2:end);
sx = [-tmp(end:-1:1) 0 tmp];
tmp = 0:opts.stride(2):searchwin(2);
tmp = tmp(2:end);
sy = [-tmp(end:-1:1) 0 tmp];

if opts.transform
  Tf=zeros(Nx,Ny,2*Nc,NI,'like',f);
  Tf(:,:,1:Nc,:)=misc.shift(f,[-1,0],'symmetric')-f;
  Tf(:,:,Nc+1:2*Nc,:)=misc.shift(f,[0,-1],'symmetric')-f;
  %Tf = sqrt(Tf(:,:,1:Nc,:).^2 + Tf(:,:,Nc+1:end,:).^2);
  f=Tf;
  Nc=2*Nc;
  clear Tf
end

ctr=1;
for kx=sx
  for ky=sy
    if (kx == 0 && ky == 0)% We do not check the (0,0)-offset case since in
      %this case each patch is compared to itself and the distance would be
      %zero. We add this weight at the end. (This is why we redefine above
      %K as K=K-1.)
      continue
    elseif (abs(kx) <= opts.excludeFromSearch(1) && abs(ky) <= opts.excludeFromSearch(2))
      continue    
    else      
      %fs=shift_inf(f,-[kx,ky]); %fs(m,n)=f(m+kx,n+ky)
      % Note that we use the shift function with inf boundary conditions so
      % that the comparison between valid and non-valid patches (those
      % which do not exist in the image domain) to give us an inf distance
      % measure. 
      
      if isequal(opts.patchDist,'euclidean')
        E=(f-misc.shift_inf(f,-[kx ky])).^2; % E=(f-fs).^2
      elseif isequal(opts.patchDist,'abs')
        E=abs(f-misc.shift_inf(f,-[kx ky])); % E=abs(f-fs)
      else
        error('knnpatch:InvalidInput','Unknown type of patch distance.');
      end
      
      % Separable filtering will speed-up computations.
      %D=vl_nnconv(E,repmat(opts.Wx,[1 1 1 Nc]),[],'stride',[opts.stride(1),1],opts.cuDNN);
      %D=vl_nnconv(D,repmat(opts.Wy,[1 1 1 Nc]),[],'stride',[1,opts.stride(2)],opts.cuDNN);
      %D=sum(D,3);
      D=vl_nnconv(E,repmat(opts.Wx,[1 1 Nc]),[],'stride',[opts.stride(1),1],opts.cuDNN);
      D=vl_nnconv(D,opts.Wy,[],'stride',[1,opts.stride(2)],opts.cuDNN);   
      D=reshape(D,[patch_num,NI]);
           
      if ctr <= opts.K
        knn(:,ctr,:)=repmat(T+ky*Nx+kx,[1,NI]); % coordinates of the center 
        %of the neighboring patch (m+kx,n+ky).
        %
        % m=rem(T-1,Nx)+1 returns the row where the pixel is stored
        % n=floor((T-1)/Nx)+1 returns the column where the pixel is stored.
        %
        % T2=(m+kx,n+ky)= (m+kx) + Nx*(n+ky-1) 
        %   =(rem(T-1,Nx)+1+kx) + Nx*(floor((T-1)/Nx)+1+ky-1)
        %   = [rem(T-1,Nx)+1+Nx*(floor((T-1)/Nx)] + kx+ky*Nx
        %   = T + kx+ky*Nx 
        %
        %  Let T-1 = i*Nx+d (where 0 <= d < Nx and i,d are integers) => 
        %  T=i*Nx+d+1. (We assume that T-1 >= 0. Otherwise the following 
        %  analysis does not hold. However, this does not create any 
        %  problems because negative coordinates do not exist).
        %
        %  Therefore,
        %  rem(T-1,Nx)=d and floor((T-1)/Nx)=floor(i+d/Nx)=i.
        %  
        %  Finally,
        %  rem(T-1,Nx)+1+Nx*(floor((T-1)/Nx)=d+1+Nx*i=T.        
        
        knn_D(:,ctr,:)=D; % Distance between the patches.         
      else
        % Check if the maximum element of knn_D is greater than the value of D
        % If yes, we keep the minimum weight. At the end we will have kept
        % the K smallest weights.
        [M,idx]=max(knn_D,[],2); % extract the maximum distance.
        idx=squeeze(idx);
        idx_=repmat((1:patch_num)',[1 NI]) + patch_num*(idx-1) + ...
         patch_num*opts.K*repmat(0:NI-1,[patch_num,1]);
        
        M=squeeze(M);
        mask=reshape(M > D, [patch_num,1,NI]);
        M=min(M,D);
        knn_D(idx_)=M;
        
        idx_(mask==0)=[]; % We store the new coordinates only for those
        % positions where M > D.
        R=repmat(T+ky*Nx+kx,[1,NI]);
        knn(idx_)=R(mask==1);
      end
      ctr=ctr+1;     
    end
  end
end

% We add the weight for the (0,0)-offset case and the corresponding
% coordinates.
V=repmat(T,[1,1,NI]);
knn=cat(2,V,knn);
V=zeros(patch_num,1,NI,'like',knn_D);
knn_D=cat(2,V,knn_D);

if nargout == 3
  LUT=zeros(Nx*Ny,1,'uint32');
  for k=1:patch_num
    LUT(T(k))=k;
  end
else
  LUT=[];
end

if opts.sorted
  [knn_D,idx]=sort(knn_D,2);
  T=repmat(repmat(uint32(1:patch_num)',[1 opts.K+1]),[1 1 NI]);
  knn=knn(T+uint32(idx-1)*patch_num+...
    patch_num*(opts.K+1)*reshape(kron(uint32(0:NI-1),ones(patch_num,opts.K+1,'uint32')),[patch_num, opts.K+1, NI]));
end
  