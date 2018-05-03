function [idx, n, I]=DerFMapNLSumT_helper(C)

%                   Inputs  
% C  : array [H W Nb NI] which holds the coordinates for the
% centers of the patches.  (H:height of the patch, W: width of the patch,
% NI: number of images);
% Nb : scalar which specifies the number of neighbors for each patch.
% 
%                   Outputs 
% idx : array [H*W*Nb, NI] 
% n   : array [H*W, Nb, NI] ( n=histc(C(:),1:H*W*NI); ). n(i) indicates how many
% times the patch_center i=(row,col,channel,image) appears in C.
% I   : array [H*W, Nb, NI] ( I=cumsum(n); I=[1; I(1:end-1)+1]; ). I(i) 
% indicates the position of the first appearance of the patch_center with
% coordinates i=(row,col,image).

[H,W,Nb,NI] = size(C);
useGPU = isa(C,'gpuArray');
if useGPU
  type = classUnderlying(C);
  colon_ = @gpuArray.colon;
else
  type = class(C);
  colon_ = @colon;
end
n = uint32(reshape(hist(reshape(C,H*W,Nb,NI),cast(colon_(1,H*W),'single')),H*W,Nb,NI));
I = cumsum(n,1); I = cat(1,ones(1,Nb,NI,'like',n),I(1:end-1,:,:)+1);
[~,idx] = sort(reshape(C,H*W*Nb,NI),1);

I1 = uint32(rem(idx-1,H*W)+1);
[~,idx] = sort(floor((idx-1)/(H*W))+1,1);
idx = bsxfun(@times,ones(H*W*Nb,NI,'like',C),cast(colon_(0,NI-1)*H*W*Nb,type)) + cast(idx,type);
idx = reshape(I1(idx),H*W,Nb,NI);
