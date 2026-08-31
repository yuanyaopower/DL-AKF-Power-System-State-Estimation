function [noisyData, sigma] = timevaryingnoise(data, sigma_rel)
% timevaryingnoise
% Add time-varying Gaussian noise whose standard deviation is proportional
% to the magnitude of each measurement sample.

sigma = sigma_rel * abs(data);

eps_noise = sigma .* randn(size(data));
noisyData = data + eps_noise;

% Retain the original numerical floor in the returned sigma.
sigma(sigma == 0) = 1e-4;

end
