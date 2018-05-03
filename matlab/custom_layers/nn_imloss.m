function Y = nn_imloss(X,Xgt,dzdy,varargin)
%NN_IMLOSS NN image loss.
%   Y = NN_IMLOSS(X, Xgt) computes the loss incurred by the estimated
%   images X given the ground-truth images Xgt.
%
%   The estimated images X and the ground-truth images Xgt are organised as 
%   image fields represented by H x W x D x N arrays. The first two 
%   dimensions, H and W, are spatial and correspond to the height and width 
%   of the field; the third dimension D is the number of image channels 
%   (for grayscale D=1 while for RGB D=3); finally, the dimension N is the 
%   number of images packed in the array.
%
%   If loss_type = 'psnr'
%
%   The loss function is defined as the negative psnr:
%     N
%   -Sum 20*log10(R*sqrt(K)/norm(X_n-Xgt_n)),
%    n=1
%
%   If loss_type = 'l1'
%
%   The loss function is defined as:
%     N
%   -Sum abs(X_n-Xgt_n)/K, 
%    n=1
%
%   where K=pixels x channels (number of spatial pixels times image
%   channels) and R is the maximum pixel intensity level.
%
%   DZDX = NN_IMLOSS(X, Xgt, DZDY) computes the derivative of the block
%   projected onto the output derivative DZDY. DZDX and DZDY have the
%   same dimensions as X and Y respectively.
%
%   NN_IMLOSS(...,'OPT', VALUE, ...) supports the additional option:
%
%   PeakVal:: 255
%     Allows to define a different maximum pixel intensity level.
%
%   Loss_TYPE:: 'psnr' | 'l1'


% stamatis@math.ucla.edu, 07/01/2016

opts.peakVal=255;
opts.loss_type = 'psnr';
opts.mode = 'normal';
opts=vl_argparse(opts,varargin);

sz = size(X);
D = X-Xgt;

if numel(sz)~=4
  sz=[sz ones(1,4-numel(sz))];
end
K=prod(sz(1:3));


% % This way during training we observe the PSNR improvement but actually we
% % can train the model's parameters either with the psnr or the l1 loss
% % functions.
% if nargin <= 2 || isempty(dzdy)
%   normF = sqrt(sum(reshape(D.*D,K,sz(4)),1));
%   Y=-20*sum(log10(opts.peakVal*sqrt(K)./normF));
% else
%   switch opts.loss_type
%     case 'psnr'
%       normF = sqrt(sum(reshape(D.*D,K,sz(4)),1));
%       dot_div=@(X,Y) X./Y;
%       Y=dzdy*(20/log(10))*bsxfun(dot_div,D,shiftdim(normF.^2,-2));
%     case 'l1'
%       Y = (dzdy*sign(D))/K;
%     otherwise
%       error('ploss :: Unknown loss function.');
%   end
% end
      
switch opts.loss_type  
  case 'psnr'    
    normF = sqrt(sum(reshape(D.*D,K,sz(4)),1));
    
    if nargin <= 2 || isempty(dzdy)
      Y=-20*sum(log10(opts.peakVal*sqrt(K)./normF));
    else
      dot_div=@(X,Y) X./Y;
      Y=dzdy*(20/log(10))*bsxfun(dot_div,D,shiftdim(normF.^2,-2));
    end
  case 'l1'
    if nargin <= 2 || isempty(dzdy)
      if isequal(opts.mode,'normal')
        Y = sum(abs(reshape(D,[],1)))/K;
      else % In the test mode we compute the PSNR
        normF = sqrt(sum(reshape(D.*D,K,sz(4)),1));
        Y=-20*sum(log10(opts.peakVal*sqrt(K)./normF));
      end        
    else
      Y = (dzdy.*sign(D))/K;
    end   
  otherwise
    error('ploss :: Unknown loss function.');
end
    