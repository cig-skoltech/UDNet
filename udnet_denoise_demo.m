function [psnr_measure,x,y,f] = udnet_denoise_demo(varargin)

run(fullfile(fileparts(mfilename('fullpath')), ...
  'matlab', 'vl_layers', 'vl_setupnn.m'));

opts.filePath = 'datasets/BSDS500/gray/102061.jpg'; % Path for the ground-truth image.
opts.noise_std = 25; % Standard deviation of the noise degrading the image.
opts.randn_seed = 19092015; % Seed for the random generator.
opts.model = 'unet'; % or 'unlnet'
opts.cuda = false;
opts = vl_argparse(opts,varargin);

f = single(imread(opts.filePath));

% Initialize the seed for the random generator
s = RandStream('mt19937ar','Seed',opts.randn_seed);
RandStream.setGlobalStream(s);

% The degraded input that we feed to the network and we want to
% reconstruct.
y = f + opts.noise_std * randn(size(f),'like',f);

if opts.cuda
  y = gpuArray(y);
  f = gpuArray(f);
end

% Run the UDNet network

if isequal(opts.model,'unet')
  x = unet_den(y,opts.noise_std);
elseif isequal(opts.model,'unlnet')
  x = unlnet_den(y,opts.noise_std);
end



psnr_measure =  misc.psnr(x,f,255);

