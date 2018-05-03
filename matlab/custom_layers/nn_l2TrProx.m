function [y,dzda] = nn_l2TrProx(x,d,stdn,alpha,dzdy,varargin)

%L2Prox layer
%   Y = NN_L2TRPROX(X,D,EPSILON) computes the Proximal map layer for the 
%   indicator function :
%
%                      { 0 if ||X-D|| <= EPSILON
%   i_C(D,EPSILON){X}= {
%                      { +inf if ||X-D|| > EPSILON
%
%   X, D and Y are of size H x W x K x N, and EPSILON = exp(ALPHA)*V*STDN 
%   is a scalar or a 1 x N vector, where V = sqrt(H*W*K-1).
%  
%   Y = D + K* (X-D) where K = EPSILON / max(||X-D||,EPSILON);
%
%   DZDX = NN_L2TRPROX(X,D,STDN,ALPHA,DZDY) computes the 
%   derivatives of the block projected onto DZDY. DZDX has the same 
%   dimensions as X.
%    
%   DZDX = K ( I - (X-D)*(X-D)^T/ max(||X-D||,EPSILON)^2) * R) * DZDY
%   
%   where R = (sgn(||X-D||-epsilon)+1)/2
%
%   DZDA = B*(X-D)^T*DZDY
%
%   where B = [ EPSILON *{ 2*max(||X-D||,EPSILON)-
%   EPSILON*(1-sgn(||X-D||-EPSILON)) } ] / [ 2*max(||X-D||,EPSILON)^2 ]
%   
% s.lefkimmiatis@skoltech.ru, 22/11/2016.

opts.derParams = true;
opts = vl_argparse(opts,varargin);

assert(isscalar(alpha),'alpha needs to be a scalar');
assert(all(stdn > 0),'NN_l2TrProx:: The noise std should be positive.');

[Nx,Ny,Nc,NI] = size(x);
numX = sqrt(Nx*Ny*Nc-1);
if isscalar(stdn)
  epsilon = ones(1,1,1,NI,'like',x)*exp(alpha)*stdn*numX;
else
  epsilon = exp(alpha)*stdn*numX;
end

epsilon = reshape(epsilon,1,1,1,NI);

xmd = x-d;
xmd_norm = sqrt(sum(sum(sum(xmd.^2,3),2),1));
max_norm = max(xmd_norm,epsilon);

if nargin < 5 || isempty(dzdy)
  dzda = [];
  y = d + bsxfun(@times,xmd,epsilon./max_norm);
else
  
  if opts.derParams
    % Simplified computation of k. The result is exactly the same in
    % both cases    
    %k = epsilon.*(2*max_norm-epsilon.*(1-signum(xmd_norm-epsilon)))./(2*max_norm.^2);
    k = epsilon./xmd_norm; k(k >= 1) = 0;
    dzda = xmd.*bsxfun(@times,k,dzdy);
    dzda = sum(dzda(:));
  else
    dzda = [];
  end   
    
  r = (signum(xmd_norm-epsilon)+1)/2;
  r = r./(max_norm.^2);
    
  y = bsxfun(@times,dzdy,epsilon./max_norm);
  dzdy = bsxfun(@times,y,r);
  %ip = reshape(sum(sum(sum(xmd.*dzdy,3),2),1),1,1,1,NI);
  ip = sum(sum(sum(xmd.*dzdy,3),2),1);
  y = y - bsxfun(@times,xmd,ip);
end


function s = signum(x)

s = -ones(size(x),'like',x);
s(x > 0) = 1;