function [widx, n, I]=FMapNLSumT_helper(C)

%                   Inputs  
% C  : array [H W Nb NI] which holds the coordinates for the
% centers of the patches.  (H:height of the patch, W: width of the patch,
% NI: number of images);
% Nb : scalar which specifies the number of neighbors for each patch.
% 
%                   Outputs 
% widx: array [H*W*Nb*NI 1] ( [~,idx]=sort(C(:)); )
% idx : array [H*W*Nb*NI 2] 
% n   : array [H W NI] ( n=histc(C(:),1:H*W*NI); ). n(i) indicates how many
% times the patch_center i=(row,col,channel,image) appears in C.
% I   : array [H W NI] ( I=cumsum(n); I=[1; I(1:end-1)+1]; ). I(i) 
% indicates the position of the first appearance of the patch_center with
% coordinates i=(row,col,image).

[H,W,Nb,NI] = size(C);
useGPU = isa(C,'gpuArray');
if useGPU
  type = classUnderlying(C);
else
  type = class(C);
end
T = misc.gen_idx_4shrink([H*W,Nb,NI],type,useGPU);
T = T(:);
C=C(:)+(T-1)*H*W;
[~,widx]=sort(C);
widx=uint32(widx);
n=uint32(histc(C,1:H*W*NI));
I=uint32(cumsum(n)); I=[1; I(1:end-1)+1];
%idx = [rem(widx-1,H*W)+1 T];%floor(widx-1,H*W*Nb)+1];

%NL_sum matlab version
% e=zeros(H,W,D,NI,'like',f);
% for i=1:(numel(idx)/Nb)
%   e(i)=sum(y(idx(I(i):I(i)+n(i)-1)));
% end
