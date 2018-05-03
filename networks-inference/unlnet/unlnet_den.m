function [out,Nbrs_idx] = unlnet_den(y,stdn,varargin)

opts.netPath = fullfile(fileparts(mfilename('fullpath')),'models');
opts.fsize = 5; % The support of the filters
opts.idx = [];
opts = vl_argparse(opts,varargin);

useColor = (size(y,3) == 3);

if useColor, opts.fsize = 5; else, opts.fsize = 7; end

if mean(stdn(:)) >= 30, std_str = 'HN'; else, std_str = 'LN'; end
if useColor, std_str = ['c' std_str]; end
model_name = sprintf('unlnet_%s_%.0dx%.0dj.mat',std_str,opts.fsize(1),opts.fsize(1));
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
  
 
if isempty(opts.idx)
  %Nc = size(y,3);
  Params.Nbrs_idx = misc.patchMatch(nn_pad(y,net.layers{1}.padSize), ...
    'stride',net.layers{1}.stride,'Nbrs',net.layers{1}.Nbrs, ...
    'searchwin',[15 15],'patchsize',size(net.layers{1}.weights{1}(:,:,1,1)));
else
  Params.Nbrs_idx = opts.idx;
  opts.idx = [];
end

if isa(y,'gpuArray')
  net = net_mv2dest(net,'gpu');
  Params = misc.move_data('gpu',Params);
end

out = unlnet_eval(net,y,[],[],'netParams',Params,'conserveMemory',true);
out = out(end).x;

Nbrs_idx = [];
if nargout > 1
  Nbrs_idx = Params.Nbrs_idx;
end
