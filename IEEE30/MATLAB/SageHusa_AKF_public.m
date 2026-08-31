function [U_fil, th_fil, rmse_u_aekf, rmse_th_aekf] = ...
    SageHusa_AKF_public(U, theta, P, Qp, case_id, noise_cfg)
% SageHusa_AKF_public
% IEEE 30-bus Sage-Husa adaptive KF comparison baseline.
%
% Inputs:
%   U, theta, P, Qp
%       Evaluation-period voltage magnitude, phase angle, active-power
%       injection, and reactive-power injection trajectories.
%
%   case_id
%       1 - stationary Gaussian noise
%       2 - Gaussian mixture noise
%       3 - time-varying Gaussian noise
%       4 - time-varying Gaussian noise + bad data at P5/P11
%
%   noise_cfg
%       Only required for Case 2. It must contain:
%           noise_cfg.sigma_P_case2
%           noise_cfg.sigma_Q_case2
%       These vectors specify the nominal P/Q measurement-noise standard
%       deviations for the Gaussian-mixture noise case.
%
% Outputs:
%   U_fil, th_fil
%       Sage-Husa AKF estimates over the supplied evaluation-period data.
%   rmse_u_aekf, rmse_th_aekf
%       Mean RMSE values after excluding the first 96 samples.
%
% Notes:
%   - MATPOWER is required for the IEEE 30-bus system model.
%   - The helper functions formHe, cal_hx, timevaryingnoise, and
%     forceSPD should be available on the MATLAB search path.

if nargin < 5
    error('U, theta, P, Qp, and case_id are required.');
end

if nargin < 6 || isempty(noise_cfg)
    noise_cfg = struct();
end

if ~ismember(case_id, 1:4)
    error('case_id must be 1, 2, 3, or 4.');
end

if size(U, 2) ~= size(theta, 2) || ...
        size(U, 2) ~= size(P, 2) || ...
        size(U, 2) ~= size(Qp, 2)
    error('U, theta, P, and Qp must have the same number of time samples.');
end

%% System model
mpc = loadcase('case30');

slack = 1;
numBus = size(mpc.bus, 1);
numbranch = size(mpc.branch, 1);

branch = mpc.branch;
branch = [branch(:, 9), branch(:, 1:5)];

for i = 1:numbranch
    if branch(i, 1) ~= 0
        branch(i, 6) = branch(i, 1);
        branch(i, 1) = 1;
    end
end

[Y, ~, ~] = makeYbus(mpc);
Y = full(Y);
G = real(Y);
B = imag(Y);

% Equivalent branch parameters:
% col 1: branch flag (0 line, 1 transformer)
% col 2-7: from bus, to bus, r, x, charging susceptance, tap ratio
% col 8-13: gij, bij, gsi, bsi, gsj, bsj
line_equ = zeros(numbranch, 13);
line_equ(:, 1:6) = branch;

for ii = 1:numbranch
    if line_equ(ii, 1) == 0
        line_equ(ii, 7) = 1;
    else
        line_equ(ii, 7) = line_equ(ii, 6);
        line_equ(ii, 6) = 0;
    end
end

for ii = 1:numbranch
    r = line_equ(ii, 4);
    x = line_equ(ii, 5);
    y = line_equ(ii, 6);
    k = line_equ(ii, 7);

    g = r / (r * r + x * x);
    b = -x / (r * r + x * x);

    line_equ(ii, 8)  = g / k;
    line_equ(ii, 9)  = b / k;
    line_equ(ii, 10) = g * (1 - k) / (k^2);
    line_equ(ii, 11) = b * (1 - k) / (k^2) + y / 2;
    line_equ(ii, 12) = g * (k - 1) / k;
    line_equ(ii, 13) = b * (k - 1) / k + y / 2;
end

%% Evaluation-period dimensions
gen = mpc.gen;
gen_bus = gen(:, 1);

U_idx = setdiff(1:numBus, gen_bus)';
theta_idx = setdiff(1:numBus, slack)';

num_U = numel(U_idx);
num_theta = numel(theta_idx);
num_x = num_U + num_theta;
col = size(U, 2);

if col < 97
    error('At least 97 evaluation-period samples are required.');
end

%% Case-1 nominal R0 used by all four cases
delta = 1e-3;

sigma_P_case1 = 1e-2 * mean(abs(P), 2);
sigma_P_case1(sigma_P_case1 == 0) = 1e-4;

sigma_Q_case1 = 1e-2 * mean(abs(Qp), 2);
sigma_Q_case1(sigma_Q_case1 == 0) = 1e-4;

R0 = diag([ ...
    delta.^2 * ones(num_U, 1); ...
    sigma_P_case1.^2; ...
    sigma_Q_case1.^2 ...
]);

%% Measurements for the selected case
switch case_id

    case 1
        % Case 1: stationary Gaussian noise
        rng(42, 'twister');

        U_z = U .* (1 + normrnd(0, delta, [numBus, col]));
        P_z = P + sigma_P_case1 .* normrnd(0, 1, [numBus, col]);
        Q_z = Qp + sigma_Q_case1 .* normrnd(0, 1, [numBus, col]);

        z = [U_z(U_idx, :); P_z; Q_z];

    case 2
        % Case 2: three-component Gaussian mixture noise
        % normal / mid / big probabilities: 0.70 / 0.20 / 0.10
        % variance multipliers: 1 / 2.25 / 25
        %
        p_big = 0.10;
        p_mid = 0.20;

        scale_mid = 2.25;
        scale_big = 25.0;

        sigma_U_h = delta;

        if ~isfield(noise_cfg, 'sigma_P_case2') || ...
                ~isfield(noise_cfg, 'sigma_Q_case2')
            error(['Case 2 requires noise_cfg.sigma_P_case2 and ' ...
                   'noise_cfg.sigma_Q_case2.']);
        end

        sigma_P_h = noise_cfg.sigma_P_case2(:);
        sigma_Q_h = noise_cfg.sigma_Q_case2(:);

        if numel(sigma_P_h) ~= numBus || numel(sigma_Q_h) ~= numBus
            error('Case-2 P/Q noise-scale vectors must have numBus elements.');
        end

        rng(2027, 'twister');

        % One shared mixture state at each time instant.
        u_h = rand(1, col);

        state_h = zeros(1, col);
        state_h(u_h >= p_big & u_h < p_big + p_mid) = 1;
        state_h(u_h < p_big) = 2;

        std_factor_h = ones(1, col);
        std_factor_h(state_h == 1) = sqrt(scale_mid);
        std_factor_h(state_h == 2) = sqrt(scale_big);

        % Generate only the voltage channels that are actually measured.
        U_z_h = U(U_idx, :) ...
            + sigma_U_h .* std_factor_h ...
            .* randn(num_U, col);

        P_z_h = P ...
            + sigma_P_h .* std_factor_h ...
            .* randn(numBus, col);

        Q_z_h = Qp ...
            + sigma_Q_h .* std_factor_h ...
            .* randn(numBus, col);

        z = [U_z_h; P_z_h; Q_z_h];

        fprintf('Case 2 noise-state ratios:\n');
        fprintf('  normal = %.4f\n', mean(state_h == 0));
        fprintf('  mid    = %.4f\n', mean(state_h == 1));
        fprintf('  big    = %.4f\n', mean(state_h == 2));

    case {3, 4}
        % Cases 3 and 4 use exactly the same time-varying Gaussian
        % realization before the intentional bad-data modification.
        rng(2028, 'twister');

        [U_z_mix, ~] = timevaryingnoise(U, 1e-3);
        [P_z_mix, ~] = timevaryingnoise(P, 1e-2);
        [Q_z_mix, ~] = timevaryingnoise(Qp, 1e-2);

        if case_id == 4
            % Persistent bad data: set P measurements at buses 5 and 11
            % to zero over the entire evaluation period.
            bad_P_buses = [5, 11];
            P_z_mix(bad_P_buses, :) = 0;
        end

        z = [U_z_mix(U_idx, :); P_z_mix; Q_z_mix];
end

len_z = size(z, 1);

%% True state
x_true = [theta(theta_idx, :); U(U_idx, :)];

%% Sage-Husa AKF parameters
x0 = x_true(:, 1);

P0 = 1e-24 * eye(num_x);
Q0 = 1e-3^2 * eye(num_x);

alpha_q = 0.02;
alpha_r = 0.02;

min_eig = 1e-12;
use_joseph = true;

% Holt local state-transition model
x_1 = 0.8;
y_1 = 0.5;

c1 = x_1 * (1 + y_1);
F = diag(c1 * ones(num_x, 1));

a = zeros(num_x, col);
b = zeros(num_x, col);

a(:, 1) = x_true(:, 1);
b(:, 1) = x_true(:, 2) - x_true(:, 1);

Ffun = @(x) F;
hfun = @(x) cal_hx( ...
    x, G, B, line_equ, numbranch, numBus, slack, ...
    U_idx, theta_idx, num_U, num_theta ...
);
Hfun = @(x) formHe( ...
    x, G, B, numBus, numbranch, line_equ, slack, ...
    U_idx, theta_idx, num_U, num_theta, gen_bus ...
);

%% Initialization
n = num_x;
m = len_z;
I = eye(n);

x_hat = zeros(n, col);
P_hat = zeros(n, n, col);
Q_hist = zeros(n, n, col);
R_hist = zeros(m, m, col);
innov_all = zeros(m, col);

x_post = x0;
P_post = P0;
Qk = forceSPD(Q0, min_eig);
Rk = forceSPD(R0, min_eig);

x_hat(:, 1) = x_post;
P_hat(:, :, 1) = P_post;
Q_hist(:, :, 1) = Qk;
R_hist(:, :, 1) = Rk;

N_burn = 100;
W = 100;

innov_buf = zeros(m, W);
w_buf = zeros(n, W);
d_buf = zeros(n, W);
buf_cnt = 0;

r_floor = 1e-8;
q_floor = 1e-10;
q_ceil = 1e-2;

%% Adaptive filtering
tic
for k = 2:col

    % 1. Prior state and covariance
    Fk = Ffun(x_post);
    x_pred = a(:, k - 1) + b(:, k - 1);
    P_pred = Fk * P_post * Fk' + Qk;

    % 2. Measurement prediction
    hk = hfun(x_pred);
    Hk = Hfun(x_pred);

    innov = z(:, k) - hk;
    S = Hk * P_pred * Hk' + Rk;

    % 3. Kalman gain
    K = (P_pred * Hk') / S;

    % 4. Posterior state and covariance
    x_post = x_pred + K * innov;

    if use_joseph
        P_post = ...
            (I - K * Hk) * P_pred * (I - K * Hk)' ...
            + K * Rk * K';
    else
        P_post = P_pred - K * S * K';
    end

    % 5. Sage-Husa adaptive Q/R update
    w_k = x_post - x_pred;

    idxb = mod(k - 2, W) + 1;
    innov_buf(:, idxb) = innov.^2;
    w_buf(:, idxb) = w_k.^2;
    d_buf(:, idxb) = diag( ...
        P_post - Fk * P_hat(:, :, k - 1) * Fk.' ...
    );

    buf_cnt = min(buf_cnt + 1, W);

    if k > N_burn
        innov_mean = mean(innov_buf(:, 1:buf_cnt), 2);
        w_mean = mean(w_buf(:, 1:buf_cnt), 2);
        d_mean = mean(d_buf(:, 1:buf_cnt), 2);

        % R update
        HPH_diag = diag(Hk * P_pred * Hk.');
        deltaR = innov_mean - HPH_diag;

        R_diag = diag(Rk);
        R_diag = ...
            (1 - alpha_r) .* R_diag ...
            + alpha_r .* max(deltaR, r_floor);
        R_diag = max(R_diag, r_floor);
        Rk = diag(R_diag);

        % Q update
        Q_diag = diag(Qk);
        Q_diag = ...
            (1 - alpha_q) .* Q_diag ...
            + alpha_q .* (w_mean + d_mean);
        Q_diag = min(max(Q_diag, q_floor), q_ceil);
        Qk = diag(Q_diag);
    end

    % Holt update
    a(:, k) = x_1 * x_post + (1 - x_1) * x_pred;
    b(:, k) = ...
        y_1 * (a(:, k) - a(:, k - 1)) ...
        + (1 - y_1) * b(:, k - 1);

    % Store
    x_hat(:, k) = x_post;
    P_hat(:, :, k) = P_post;
    Q_hist(:, :, k) = Qk;
    R_hist(:, :, k) = Rk;
    innov_all(:, k) = innov;
end
toc

%% Results
th_fil = x_hat(1:num_theta, :);
U_fil = x_hat(num_theta + 1:end, :);

% Exclude the first 96 warm-up samples from the RMSE calculation.
eval_start = 97;

rmse_u_all = sqrt(mean( ...
    (U(U_idx, eval_start:end) - U_fil(:, eval_start:end)).^2, 1 ...
))';

rmse_th_all = sqrt(mean( ...
    (theta(theta_idx, eval_start:end) - th_fil(:, eval_start:end)).^2, 1 ...
))';

rmse_u_aekf = mean(rmse_u_all, 1);
rmse_th_aekf = mean(rmse_th_all, 1);

fprintf('Case %d: RMSE_U = %.6e, RMSE_theta = %.6e\n', ...
    case_id, rmse_u_aekf, rmse_th_aekf);

end
