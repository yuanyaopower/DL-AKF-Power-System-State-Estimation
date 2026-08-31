function [noisyData, sigma] = timevaryingnoise(data, sigma_rel)
% Add time-varying Gaussian measurement noise with signal-dependent scale.

sigma = sigma_rel * abs(data);

eps_noise = sigma .* randn(size(data));
noisyData = data + eps_noise;

% Preserve channels that are treated as strict zero injections.
noisyData(data(:, 1) == 0, :) = 0;

% Apply a numerical floor to zero entries in the returned sigma.
sigma(sigma == 0) = 1e-4;

end
