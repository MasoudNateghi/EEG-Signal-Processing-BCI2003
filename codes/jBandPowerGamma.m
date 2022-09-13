
function BPG = jBandPowerGamma(X,opts)
% Parameters         
f_low  = 30;      % 30 Hz
f_high = 50;      % 50 Hz

% sampling frequency 
if isfield(opts,'fs'), fs = opts.fs; end

% Band power 
BPG = bandpower(X, fs, [f_low f_high]);
end

