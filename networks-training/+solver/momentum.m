function [w, state] = momentum(w, state, grad, opts, lr)
%ADAGRAD
%   Example Momentum SGD solver, for use with CNN_TRAIN and CNN_TRAIN_DAG.
%
%   If called without any input argument, returns the default options
%   structure.
%
%   Solver options: (opts.train.solverOpts)
%
%   `gamma`:: 0.9
%      Momentum coefficient must be in the range [0,1]
%
%   `decay`:: 5e-4
%      Decay rate

if nargin == 0  % Return the default solver options
  w = struct('gamma', 0.9, 'decay', 0) ;
  return ;
end

state = opts.gamma*state - opts.decay*lr*w - lr*grad;
w = w + state;
