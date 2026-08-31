function [noisyData, sigma] = timevaryingnoise(data, sigma_rel)
% Add time-varying Gaussian measurement noise with signal-dependent scale.

sigma = sigma_rel * abs(data);

eps_noise = sigma .* randn(size(data));
noisyData = data + eps_noise;

end
