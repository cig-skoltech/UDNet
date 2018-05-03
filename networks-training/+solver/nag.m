function [w, state] = nag(w, state, grad, opts, lr)
%ADAGRAD
%   Example Nesterov's Accelerated Gradient Solver, for use with CNN_TRAIN 
%   and CNN_TRAIN_DAG.
%
%   If called without any input argument, returns the default options
%   structure.
%
%   Solver options: (opts.train.solverOpts)
%
%   `gamma`:: 0.9
%      Momentum coefficient must be in the range [0,1]
%
% See http://cs231n.github.io/neural-networks-3/#sgd


if nargin == 0  % Return the default solver options
  w = struct('gamma', 0.9) ;
  return ;
end

state_prev = state;
state = opts.gamma*state - lr*grad;
w = w + state + opts.gamma*(state-state_prev);


