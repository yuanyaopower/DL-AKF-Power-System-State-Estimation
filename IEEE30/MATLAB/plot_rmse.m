% plot_rmse.m
% Public IEEE-30 plotting script.
%
% Input:
%   ../Results/rmse_all_cases.mat
%
% Run aggregate_results.m first.

clc;
clear;

%% Path
script_dir = fileparts(mfilename('fullpath'));
case_dir = fileparts(script_dir);
results_dir = fullfile(case_dir, 'Results');

summary_file = fullfile(results_dir, 'rmse_all_cases.mat');

if ~isfile(summary_file)
    error( ...
        ['RMSE summary file not found:\n%s\n' ...
         'Run aggregate_results.m first.'], ...
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

assert(numel(casesTh) == n_case, ...
    'U/theta case counts are inconsistent.');

assert(numel(case_ids) == n_case, ...
    'case_ids length does not match the RMSE case count.');

assert(numel(names) == n_case, ...
    'case_names length does not match the RMSE case count.');

for i = 1:n_case
    assert(size(casesU{i}, 2) == m, ...
        'RMSE U method count mismatch in %s.', names{i});

    assert(size(casesTh{i}, 2) == m, ...
        'RMSE theta method count mismatch in %s.', names{i});
end

%% Plot settings
colors = {'r', 'm', 'k', 'b', 'b'};
shapes = {'o', '+', 's', 'd', 'x'};
symbols = {'o', '+', 's', 'd', 'x'};

assert( ...
    numel(colors) == m && ...
    numel(shapes) == m && ...
    numel(symbols) == m, ...
    'Plot-style definitions do not match the method count.' ...
);

%% Cases I-III
main_idx = find(ismember(case_ids, [1, 2, 3]));

if ~isempty(main_idx)

    figure('Color', 'w');

    for k = 1:numel(main_idx)
        i = main_idx(k);

        subplot(1, numel(main_idx), k);
        plot_one_metric( ...
            casesU{i}, meth, colors, shapes, symbols, m ...
        );
        title(names{i});
    end

    figure('Color', 'w');

    for k = 1:numel(main_idx)
        i = main_idx(k);

        subplot(1, numel(main_idx), k);
        plot_one_metric( ...
            casesTh{i}, meth, colors, shapes, symbols, m ...
        );
        title(names{i});
    end
end

%% Optional Case IV
case4_idx = find(case_ids == 4, 1);

if ~isempty(case4_idx)

    figure('Color', 'w');

    subplot(1, 2, 1);
    plot_one_metric( ...
        casesU{case4_idx}, meth, colors, shapes, symbols, m ...
    );
    title('Case IV');

    subplot(1, 2, 2);
    plot_one_metric( ...
        casesTh{case4_idx}, meth, colors, shapes, symbols, m ...
    );
    title('Case IV');

else
    fprintf('Case IV is not included in the released summary file.\n');
end

%% Mean RMSE
mean_U = zeros(n_case, m);
mean_Th = zeros(n_case, m);

for i = 1:n_case
    mean_U(i, :) = mean(casesU{i}, 1, 'omitnan');
    mean_Th(i, :) = mean(casesTh{i}, 1, 'omitnan');
end

disp('Mean RMSE of voltage magnitude:');
disp(array2table( ...
    mean_U, ...
    'VariableNames', matlab.lang.makeValidName(meth), ...
    'RowNames', names ...
));

disp('Mean RMSE of voltage phase angle:');
disp(array2table( ...
    mean_Th, ...
    'VariableNames', matlab.lang.makeValidName(meth), ...
    'RowNames', names ...
));

%% Local function
function plot_one_metric( ...
    M, meth, colors, shapes, symbols, m ...
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
            'symbol', symbols{j}, ...
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

        legend_text{j} = sprintf('%.3e', mu);
    end

    xlim([0.5, m + 0.5]);
    xticks(1:m);
    xticklabels(meth);

    ylabel('RMSE');

    leg = legend( ...
        h_legend, ...
        legend_text, ...
        'Location', 'northwest' ...
    );

    leg.Title.String = 'Mean RMSE';
end

