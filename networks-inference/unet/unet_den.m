function out = unet_den(y,stdn,varargin)

opts.netPath = fullfile(fileparts(mfilename('fullpath')),'models');
opts.fsize = 5; % The support of the filters
opts = vl_argparse(opts,varargin);

useColor = (size(y,3) == 3);

if useColor, opts.fsize = 5; else, opts.fsize = 7; end

if stdn(1) >= 30, std_str = 'HN'; else, std_str = 'LN'; end
if useColor, std_str = ['c' std_str]; end
model_name = sprintf('unet_%s_%.0dx%.0dj.mat',std_str,opts.fsize(1),opts.fsize(1));
opts.netPath = fullfile(opts.netPath,model_name);

l=load(opts.netPath);
net = l.net;
clear l;

Params = net.meta.netParams;
Params.stdn = stdn; 

if isa(y,'gpuArray')
  cid = classUnderlying(y);
else
  cid = class(y);
end

if strcmp(cid,'double')
  for k=1:numel(net.layers)
    if isfield(net.layers{k},'weights')
      for j=1:numel(net.layers{k}.weights)
        net.layers{k}.weights{j} = double(net.layers{k}.weights{j});
      end
    end
    if isfield(net.layers{k},'rbf_means')
      net.layers{k}.rbf_means = double(net.layers{k}.rbf_means);
    end
  end
  
  Params.data_mu = double(Params.data_mu);
end

if isa(y,'gpuArray')
  net = net_mv2dest(net,'gpu');
  Params = misc.move_data('gpu',Params);
end

out = unet_eval(net,y,[],[],'netParams',Params,'conserveMemory',true);
out = out(end).x;
