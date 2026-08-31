function [U_fil, th_fil, rmse_u_ekf, rmse_th_ekf] = ...
    EKF_public(U, theta, P, Qp, case_id, noise_cfg)
% EKF_public
% IEEE 30-bus EKF comparison baseline.
%
% Inputs:
%   U, theta, P, Qp
%       Evaluation-period voltage magnitude, phase angle, active-power
%       injection, and reactive-power injection trajectories. The first
%       96 samples are used as the warm-up/history interval.
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
%       EKF estimates over the supplied evaluation-period trajectories.
%   rmse_u_ekf, rmse_th_ekf
%       Mean RMSE values after excluding the first 96 samples.
%
% Notes:
%   - MATPOWER is required for the IEEE 30-bus system model.
%   - The helper functions formHe, cal_hx, and timevaryingnoise
%     should be available on the MATLAB search path.

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

num_U = size(U_idx, 1);
num_theta = size(theta_idx, 1);
num_x = num_U + num_theta;
col = size(U, 2);

if col < 97
    error('At least 97 evaluation-period samples are required.');
end

%% Fixed R for all cases: Case-1 nominal covariance
delta = 1e-3;

sigma_P_case1 = 1e-2 * mean(abs(P), 2);
sigma_P_case1(sigma_P_case1 == 0) = 1e-4;

sigma_Q_case1 = 1e-2 * mean(abs(Qp), 2);
sigma_Q_case1(sigma_Q_case1 == 0) = 1e-4;

R_case1 = diag([ ...
    delta.^2 * ones(num_U, 1); ...
    sigma_P_case1.^2; ...
    sigma_Q_case1.^2 ...
]);

R = R_case1;

%% Measurements for the selected noise case
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
        p_big = 0.10;
        p_mid = 0.20;

        scale_mid = 2.25;   % variance multiplier
        scale_big = 25.0;   % variance multiplier

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

%% EKF setup
x_rea = [theta(theta_idx, :); U(U_idx, :)];

x_pre = zeros(num_x, col);
x_fil = zeros(num_x, col);

% Holt two-parameter prior
a = zeros(num_x, col);
b = zeros(num_x, col);

Q = diag((1e-6) * ones(num_x, 1));

x_1 = 0.8;
y_1 = 0.5;

c_1 = x_1 * (1 + y_1);
F = diag(c_1 * ones(num_x, 1));

%% Initialization
x_pre(:, 1) = x_rea(:, 1);
x_fil(:, 1) = x_rea(:, 1);

a(:, 1) = x_rea(:, 1);
b(:, 1) = x_rea(:, 2) - x_rea(:, 1);

P_fil = 1e-24 * eye(num_x);

% Time step 2
x_pre(:, 2) = a(:, 1) + b(:, 1);

P_pre = F * P_fil * F' + Q;
H = formHe( ...
    x_pre(:, 2), G, B, numBus, numbranch, line_equ, ...
    slack, U_idx, theta_idx, num_U, num_theta, gen_bus ...
);
S = H * P_pre * H' + R;
K = P_pre * H' * inv(S);

hx = cal_hx( ...
    x_pre(:, 2), G, B, line_equ, numbranch, numBus, ...
    slack, U_idx, theta_idx, num_U, num_theta ...
);
delta_z = z(:, 2) - hx;

x_fil(:, 2) = x_pre(:, 2) + K * delta_z;
P_fil = (diag(ones(num_x, 1)) - K * H) * P_pre;

a(:, 2) = x_1 * x_fil(:, 2) + (1 - x_1) * x_pre(:, 2);
b(:, 2) = y_1 * (a(:, 2) - a(:, 1)) + (1 - y_1) * b(:, 1);

%% EKF recursion
tic
for i = 3:col

    % Prior state and covariance
    x_pre(:, i) = a(:, i - 1) + b(:, i - 1);
    P_pre = F * P_fil * F' + Q;

    H = formHe( ...
        x_pre(:, i), G, B, numBus, numbranch, line_equ, ...
        slack, U_idx, theta_idx, num_U, num_theta, gen_bus ...
    );

    S = H * P_pre * H' + R;
    K = P_pre * H' * inv(S);

    % Measurement update
    hx = cal_hx( ...
        x_pre(:, i), G, B, line_equ, numbranch, numBus, ...
        slack, U_idx, theta_idx, num_U, num_theta ...
    );

    delta_z = z(:, i) - hx;
    x_fil(:, i) = x_pre(:, i) + K * delta_z;
    P_fil = (diag(ones(num_x, 1)) - K * H) * P_pre;

    % Holt update
    a(:, i) = x_1 * x_fil(:, i) + (1 - x_1) * x_pre(:, i);
    b(:, i) = y_1 * (a(:, i) - a(:, i - 1)) ...
        + (1 - y_1) * b(:, i - 1);
end
toc

%% Results
th_fil = x_fil(1:num_theta, :);
U_fil = x_fil(num_theta + 1:end, :);

% Exclude the first 96 warm-up samples from the RMSE calculation.
rmse_u_ekf = mean(sqrt(mean( ...
    (U(U_idx, 97:end) - U_fil(:, 97:end)).^2, 1 ...
)));

rmse_th_ekf = mean(sqrt(mean( ...
    (theta(theta_idx, 97:end) - th_fil(:, 97:end)).^2, 1 ...
)));

fprintf('Case %d: RMSE_U = %.6e, RMSE_theta = %.6e\n', ...
    case_id, rmse_u_ekf, rmse_th_ekf);

end
