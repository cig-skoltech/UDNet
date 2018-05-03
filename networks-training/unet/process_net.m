function net = process_net(net)

numLayers = numel(net.layers);

if isfield(net.layers{1},'precious')
  net.layers{1} = rmfield(net.layers{1},'precious');
end

for k=2:numLayers
  if isfield(net.layers{k},'first_stage')
    net.layers{k}.first_stage = false;
  end
  if isfield(net.layers{k},'precious')
    net.layers{k} = rmfield(net.layers{k},'precious');
  end  
end


