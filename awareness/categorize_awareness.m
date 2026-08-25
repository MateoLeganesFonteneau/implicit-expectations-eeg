%% Bayesian awareness categorization using all available trials
% Reproduces the final Bayesian Awareness Categorization Technique (BACT)
% analysis. The input contains one row per awareness trial:
%   column 1: participant ID
%   column 2: accuracy (0 or 1)
%   column 3: stimulus type (0 or 1)
%
% Outputs are written to awareness/outputs/ as both CSV and MAT files.

clearvars;
clc;

%% Paths
script_dir = fileparts(mfilename('fullpath'));
data_path = fullfile(script_dir, 'raw_data', 'contingency_data_EEG.mat');
output_dir = fullfile(script_dir, 'outputs');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Load and validate trial-level data
input = load(data_path, 'data');
assert(isfield(input, 'data'), 'Input file must contain a variable named data.');

data = double(input.data);
assert(size(data, 2) == 3, 'The data matrix must have exactly three columns.');
assert(all(isfinite(data), 'all'), 'The data matrix contains non-finite values.');
assert(all(ismember(data(:, 2), [0, 1])), 'Accuracy must be coded as 0 or 1.');
assert(all(ismember(data(:, 3), [0, 1])), 'Stimulus type must be coded as 0 or 1.');

%% Participant selection
exclude_ids = [1001, 1012, 1028, 1009, 1025, 1064, 1066, 1047, 1048, 1011];
data(ismember(data(:, 1), exclude_ids), :) = [];
participant_ids = unique(data(:, 1), 'sorted');
n_participants = numel(participant_ids);

%% Bayesian model settings
d1_prior_upper = 2;    % uniform prior for log d' on [0, 2]
bf_threshold = 3;      % aware >= 3; unaware < 1/3
integration_steps = 2000;
integration_increment = d1_prior_upper / integration_steps;

% This grid reproduces the numerical integration used in the original code.
theta_grid = (1:integration_steps)' .* integration_increment;
normal_pdf = @(mean_value, variance, x) ...
    exp(-((x - mean_value).^2) ./ (2 .* variance)) ./ ...
    sqrt(2 .* pi .* variance);

%% Preallocate participant-level results
n_trials = zeros(n_participants, 1);
log_d1 = nan(n_participants, 1);
se_log_d1 = nan(n_participants, 1);
bayes_factor = nan(n_participants, 1);
category_code = nan(n_participants, 1);
category = strings(n_participants, 1);

%% Categorize each participant using all available trials
for p = 1:n_participants
    participant_id = participant_ids(p);
    participant_data = data(data(:, 1) == participant_id, :);
    n_trials(p) = size(participant_data, 1);

    % Type-1 response counts
    hits = sum(participant_data(:, 2) == 1 & participant_data(:, 3) == 1);
    false_alarms = sum(participant_data(:, 2) == 0 & participant_data(:, 3) == 0);
    correct_rejections = sum(participant_data(:, 2) == 1 & participant_data(:, 3) == 0);
    misses = sum(participant_data(:, 2) == 0 & participant_data(:, 3) == 1);

    % Apply the original 0.5 correction when any response cell is empty.
    if any([hits, false_alarms, correct_rejections, misses] == 0)
        hits = hits + 0.5;
        false_alarms = false_alarms + 0.5;
        correct_rejections = correct_rejections + 0.5;
        misses = misses + 0.5;
    end

    odds_ratio = (hits / misses) / (false_alarms / correct_rejections);
    log_d1(p) = log(odds_ratio) * (sqrt(3) / pi);
    se_log_d1(p) = sqrt(1 / hits + 1 / misses + ...
        1 / false_alarms + 1 / correct_rejections) * (sqrt(3) / pi);

    variance = se_log_d1(p)^2;
    alternative_likelihood = sum(...
        (1 / d1_prior_upper) .* normal_pdf(theta_grid, variance, log_d1(p))) ...
        .* integration_increment;
    null_likelihood = normal_pdf(0, variance, log_d1(p));
    bayes_factor(p) = alternative_likelihood / null_likelihood;

    if bayes_factor(p) >= bf_threshold
        category_code(p) = 1;
        category(p) = "Aware";
    elseif bayes_factor(p) < 1 / bf_threshold
        category_code(p) = 0;
        category(p) = "Unaware";
    else
        category_code(p) = 2;
        category(p) = "Insensitive";
    end
end

%% Save results
results = table(participant_ids, n_trials, log_d1, se_log_d1, ...
    bayes_factor, category_code, category, ...
    'VariableNames', {'participant_id', 'n_trials', 'log_d1', 'se_log_d1', ...
    'bayes_factor', 'category_code', 'category'});

settings = struct( ...
    'excluded_participant_ids', exclude_ids, ...
    'd1_prior_lower', 0, ...
    'd1_prior_upper', d1_prior_upper, ...
    'bayes_factor_threshold', bf_threshold, ...
    'integration_steps', integration_steps, ...
    'trials_used', 'all available trials');

csv_path = fullfile(output_dir, 'awareness_categorization.csv');
mat_path = fullfile(output_dir, 'awareness_categorization.mat');
writetable(results, csv_path);
save(mat_path, 'results', 'settings');

fprintf('\nBayesian awareness categorization complete (N = %d).\n', n_participants);
fprintf('  Unaware:    %d\n', sum(category_code == 0));
fprintf('  Aware:      %d\n', sum(category_code == 1));
fprintf('  Insensitive:%d\n', sum(category_code == 2));
fprintf('Results saved to:\n  %s\n  %s\n', csv_path, mat_path);

