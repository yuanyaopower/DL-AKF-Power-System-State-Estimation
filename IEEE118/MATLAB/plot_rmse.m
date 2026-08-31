% Boxplot comparison for IEEE-118 simulation results
%
% Input:
%   ../Results/rmse_all_cases.mat
%
% Case order:
%   Case I  - stationary Gaussian
%   Case II - Gaussian mixture
%   Case III- time-varying Gaussian
%
% Method order:
%   EKF, AKF, DL-AKF(Holt), DL-AKF(Fixed), DL-AKF

clc;
clear;

%% Load aggregated RMSE data
script_dir = fileparts(mfilename('fullpath'));
summary_file = fullfile( ...
    script_dir, '..', 'Results', 'rmse_all_cases.mat' ...
);

if ~isfile(summary_file)
    error( ...
        ['RMSE summary file not found:\n%s\n' ...
         'Run the IEEE-118 result aggregation script first.'], ...
        summary_file ...
    );
end

S = load(summary_file);

required_vars = { ...
    'rmse_U_cases', ...
    'rmse_Th_cases', ...
    'case_ids', ...
    'case_names', ...
    'method_names' ...
};

for i = 1:numel(required_vars)
    if ~isfield(S, required_vars{i})
        error( ...
            'Variable "%s" is missing from:\n%s', ...
            required_vars{i}, summary_file ...
        );
    end
end

casesU = S.rmse_U_cases;
casesTh = S.rmse_Th_cases;

case_ids = S.case_ids;
names = S.case_names;
meth = S.method_names;

n_case = numel(casesU);
m = numel(meth);

assert(n_case == 3, ...
    'IEEE-118 comparison is expected to contain exactly three cases.');

assert(isequal(case_ids, 1:3), ...
    'IEEE-118 case order must be [1 2 3].');

assert(numel(casesTh) == n_case, ...
    'U/theta case counts are inconsistent.');

assert(numel(names) == n_case, ...
    'case_names length does not match the RMSE case count.');

for i = 1:n_case
    if size(casesU{i}, 2) ~= m || size(casesTh{i}, 2) ~= m
        error( ...
            'RMSE matrix column count does not match the method count in %s.', ...
            names{i} ...
        );
    end

    if size(casesU{i}, 1) ~= size(casesTh{i}, 1)
        error( ...
            'U/theta RMSE lengths are inconsistent in %s.', ...
            names{i} ...
        );
    end
end

%% Plot settings
colors = {'r', 'm', 'k', 'b', 'b'};
shapes = {'o', '+', 's', 'd', 'x'};
sym = {'o', '+', 's', 'd', 'x'};

if any([numel(colors), numel(shapes), numel(sym)] ~= m)
    error('Plot-style definitions do not match the method count.');
end

%% Voltage magnitude
figure('Color', 'w');

for i = 1:n_case
    subplot(1, n_case, i);
    plot_one_metric( ...
        casesU{i}, meth, colors, shapes, sym, m ...
    );
    title(names{i});
end

%% Voltage phase angle
figure('Color', 'w');

for i = 1:n_case
    subplot(1, n_case, i);
    plot_one_metric( ...
        casesTh{i}, meth, colors, shapes, sym, m ...
    );
    title(names{i});
end

%% Mean RMSE
mean_U = zeros(n_case, m);
mean_Th = zeros(n_case, m);

for i = 1:n_case
    mean_U(i, :) = mean(casesU{i}, 1, 'omitnan');
    mean_Th(i, :) = mean(casesTh{i}, 1, 'omitnan');
end

disp('Mean RMSE of U:');
disp(mean_U);

disp('Mean RMSE of theta:');
disp(mean_Th);

%% Local function
function plot_one_metric( ...
    M, meth, colors, shapes, sym, m ...
)

    hold on;
    box on;
    grid on;

    h_legend = gobjects(1, m);
    legend_text = cell(1, m);

    for j = 1:m

        pos = j;

        boxplot( ...
            M(:, j), ...
            'positions', pos, ...
            'widths', 0.6, ...
            'labels', {}, ...
            'symbol', sym{j}, ...
            'Whisker', 2 ...
        );

        mu = mean(M(:, j), 'omitnan');

        fmt = [colors{j}, shapes{j}];

        h_legend(j) = plot( ...
            pos, mu, fmt, ...
            'MarkerSize', 8, ...
            'LineWidth', 1.2, ...
            'MarkerFaceColor', 'r' ...
        );

        legend_text{j} = format_scientific(mu);
    end

    xlim([0.5, m + 0.5]);
    xticks(1:m);
    xticklabels(meth);

    ylabel('RMSE');

    leg = legend(h_legend, legend_text, 'Location', 'northwest');
    leg.Title.String = 'Mean RMSE';
end

function txt = format_scientific(value)

    if value == 0
        txt = '0';
        return;
    end

    exponent = floor(log10(abs(value)));
    mantissa = value / (10 ^ exponent);

    txt = sprintf('%.2fx10^{%d}', mantissa, exponent);
end


