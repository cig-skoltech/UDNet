function net = unlnet_add_layer(net,varargin)

opts.cid='single';
opts.inputSize = [180 180 1];
opts.layer_type = [];

%-% Params for layers CPGU
opts.patchSize=[5 5];
opts.numFilters = [];
opts.stride=[1,1];
opts.padSize=[];
opts.padType='symmetric';
opts.shrink_type = 'identity';
opts.h = [];
opts.ht = [];
opts.s = [];
opts.st = [];
opts.g = [];
opts.gt = [];
opts.weightSharing = false;
opts.groupWeightSharing = false;
opts.zeroMeanFilters = false;
opts.weightNormalization = false(1,2);
opts.rbf_means = -310:10:310;
opts.rbf_precision = [];
opts.rbf_weights = [];
opts.learningRate = 1;
opts.first_stage = false;
opts.alpha = [];
opts.clb = -100;
opts.cub = 100;
opts.Nbrs = [];

%-% Params for layer ploss
opts.peakVal = 255;
opts.loss_type = 'psnr';

% --- CLIP layer -----
opts.lb = 0;
opts.ub = 255;

opts.layer_id = 1;
opts = vl_argparse(opts,varargin);



if numel(opts.patchSize) < 3
  Nc = 1;
else
  Nc = opts.patchSize(3);
  opts.patchSize = opts.patchSize(1:2);
end

switch opts.layer_type
  
  case {'unlnet'}
    
    if isempty(opts.padSize)
      if opts.stride(1) == 1 && opts.stride(2) == 1
        Pc = floor((opts.patchSize+1)/2); % center of the patch
        opts.padSize = [Pc(1)-1, opts.patchSize(1)-Pc(1), ...
          Pc(2)-1 opts.patchSize(2)-Pc(2)];
      else
        opts.padSize = misc.getPadSize(opts.inputSize,opts.patchSize,opts.stride);
      end
    end
    
    if isempty(opts.numFilters)
      if isempty(opts.h)
        opts.numFilters = prod(opts.patchSize(1:2))-1;
      else
        opts.numFilters = size(opts.h,4);
      end
    end
    
    if isempty(opts.h)
      if prod(opts.patchSize)*Nc == prod(opts.numFilters)+1
        opts.h = misc.gen_dct3_kernel([opts.patchSize(:)',Nc],'classType',opts.cid);
        opts.h = opts.h(:,:,:,2:end);
      else      
        if Nc == 1
          opts.h = misc.odctndict(opts.patchSize,opts.numFilters+1);
        else
          opts.h = misc.odctndict([opts.patchSize, Nc],opts.numFilters+1);
        end
        opts.h = opts.h(:,2:opts.numFilters+1);
        opts.h = cast(opts.h,opts.cid);
        opts.h = reshape(opts.h,[opts.patchSize,Nc,opts.numFilters]);
      end
    else
      opts.h = cast(opts.h,opts.cid);
    end     
    
    if isempty(opts.ht)
      opts.ht = opts.h;
    else
      opts.ht = cast(opts.ht,opts.cid);
    end
    
    if isempty(opts.s)
      opts.s = ones(1,opts.numFilters,opts.cid);
    elseif ischar(opts.s) && strcmp(opts.s,'None')
      opts.s = [];            
    else
      opts.s = cast(opts.s,opts.cid);
    end
    
    if isempty(opts.st)
      opts.st = opts.s;
    elseif ischar(opts.st) && strcmp(opts.st,'None')
      opts.st = [];      
    else
      opts.st = cast(opts.st,opts.cid);
    end
    
    if isempty(opts.g)
      opts.g = zeros(1,opts.Nbrs,opts.cid);
      opts.g(1,1) = 1;
    else
      opts.g = cast(opts.g,opts.cid);
    end
    
    if isempty(opts.gt)
      opts.gt = opts.g;
    else
      opts.gt = cast(opts.gt,opts.cid);
    end
        
    nlr = numel(opts.learningRate);
    if nlr~=8
      opts.learningRate = [opts.learningRate(:)',ones(1,8-nlr,'like',opts.learningRate)];
    end
    
    if opts.weightSharing
      opts.ht = [];
      opts.st = [];
      opts.learningRate([4,5]) = 0; % We don't learn any independent weights for the conv2Dt layers      
    end   
    
    if opts.groupWeightSharing 
      opts.gt = [];      
      opts.learningRate(6) = 0;% We don't learn any independent weights for the FMapNLSumT layers      
    end
    
    if isempty(opts.s), opts.learningRate(2) = 0; end
    if isempty(opts.st), opts.learningRate(5) = 0; end
    
    if isempty(opts.alpha)
      opts.alpha = 0;
    end
        
    opts.rbf_means = cast(opts.rbf_means,opts.cid);
    
    if isempty(opts.rbf_precision)
      opts.rbf_precision = opts.rbf_means(2)-opts.rbf_means(1);
    end
    
    if isempty(opts.rbf_weights)
      opts.rbf_weights = 1e-4*ones(opts.numFilters,numel(opts.rbf_means),opts.cid);
    else
      opts.rbf_weights = cast(opts.rbf_weights,opts.cid);
    end
        
    net.layers{end+1} = struct('type', opts.layer_type, ...
      'name', [opts.layer_type '_' num2str(opts.layer_id)], ...
      'weights', {{opts.h, opts.s, opts.g, opts.ht, opts.st, opts.gt, ...
      opts.rbf_weights, opts.alpha}},...
      'rbf_means', opts.rbf_means, ...
      'rbf_precision', opts.rbf_precision, ...
      'stride', opts.stride, ...
      'padSize', opts.padSize, ...
      'padType', opts.padType, ...
      'Nbrs', opts.Nbrs, ...
      'weightNormalization', opts.weightNormalization, ...
      'zeroMeanFilters', opts.zeroMeanFilters, ...
      'shrink_type', opts.shrink_type, ...
      'learningRate', opts.learningRate, ...
      'lb', opts.clb, ...
      'ub', opts.cub, ...
      'first_stage', opts.first_stage);
        
  case 'clip'
    
    if ~isinf(opts.lb) || ~isinf(opts.ub)
      net.layers{end+1} = struct('type', 'clip', ...
        'name', ['proj_' num2str(opts.layer_id)], ...
        'lb', opts.lb, ...
        'ub', opts.ub) ;
    end
    
  case 'imloss'
    
    net.layers{end+1} = struct('type', 'imloss', ...
      'name', 'im_loss', 'peakVal', opts.peakVal, 'loss_type', ...
      opts.loss_type) ;
    
  otherwise
    error('Unkown layer type');
    
end
