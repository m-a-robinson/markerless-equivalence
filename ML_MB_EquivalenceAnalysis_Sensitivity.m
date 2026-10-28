%% Sensitivity analysis of equivalence bounds - ML vs MB (Study 2)
% Compares the MDC-based bounds with fixed +/-2 and +/-5 degree bounds, and
% classifies each point of the gait cycle into one of three outcomes.
%
% Requires (in the workspace, e.g. from stydy2data_20260226_withMDC.mat):
%   stats.equivalence_mdc   90% spm1d paired CIs (from ML_MB_EquivalenceAnalysis_MDC.m)
%   gait_allsubj_sw_mdc     Study 1 MDC
%
% Outcome definitions (90% CI = two one-sided tests at alpha = 0.05):
%   EQUIVALENT      CI lies entirely inside   [-bound, +bound]  (TOST rejects non-equivalence)
%   NOT EQUIVALENT  CI lies entirely outside  [-bound, +bound]  (difference reliably exceeds the bound)
%   INCONCLUSIVE    CI straddles a bound      (cannot conclude either way)

plttitles = {'Pelvis Anterior Tilt', 'Pelvis Obliquity', 'Pelvis Rotation',...
    'Hip Flexion', 'Hip Adduction', 'Hip Rotation',...
    'Knee Flexion', 'Knee Varus', 'Foot Progression',...
    'Ankle Dorsiflexion', 'Ankle Inversion', 'Ankle Rotation'};
plt_order_r = [11,12,13,18,19,20,21,22,17,14,15,16];
mdc_order   = [1,2,3,7,8,9,10,11,12,4,5,6];

%% Bound sets (one column per set)
mdc_bound = cell2mat(gait_allsubj_sw_mdc(mdc_order, 2));
mdc_bound(9) = 5;   % Foot progression: not examined in Study 1, default +/-5

bound_names = {'MDC', 'fixed2', 'fixed5'};
bounds = [mdc_bound, 2*ones(12,1), 5*ones(12,1)];   % 12 x 3
nb = numel(bound_names);

%% Classify every gait-cycle node
% class codes: 1 = equivalent, 2 = not equivalent, 3 = inconclusive
sens.class = cell(12, nb);
sens.pct   = nan(12, nb, 3);     % % of gait cycle in [equivalent, not equivalent, inconclusive]

for jj = 1:12
    ci = stats.equivalence_mdc{plt_order_r(jj), 1}.ci;   % 2 x 101
    lo = min(ci, [], 1);
    hi = max(ci, [], 1);

    for bb = 1:nb
        b = bounds(jj, bb);

        is_equiv = lo >= -b & hi <= b;
        is_not   = lo >  b  | hi < -b;

        cls = 3 * ones(1, numel(lo));    % default inconclusive
        cls(is_equiv) = 1;
        cls(is_not)   = 2;

        sens.class{jj, bb} = cls;
        for cc = 1:3
            sens.pct(jj, bb, cc) = 100 * mean(cls == cc);
        end
    end
end

%% Summary table
rows = {};
for jj = 1:12
    for bb = 1:nb
        rows(end+1,:) = {plttitles{jj}, bound_names{bb}, bounds(jj,bb), ...
            sens.pct(jj,bb,1), sens.pct(jj,bb,2), sens.pct(jj,bb,3)}; %#ok<SAGROW>
    end
end
sens.table = cell2table(rows, 'VariableNames', ...
    {'Variable','BoundSet','Bound_deg','PctEquivalent','PctNotEquivalent','PctInconclusive'});
disp(sens.table)
writetable(sens.table, 'equivalence_sensitivity_summary.csv');

% Mean across the 12 variables
for bb = 1:nb
    fprintf('%-7s mean %%: equivalent %.1f | not equivalent %.1f | inconclusive %.1f\n', ...
        bound_names{bb}, squeeze(mean(sens.pct(:,bb,:),1)));
end

%% Plot: CI with bounds for each set, coloured by outcome
cols = [0.0 0.65 0.2;     % equivalent
        0.85 0.15 0.15;   % not equivalent
        0.6 0.6 0.6];     % inconclusive

for bb = 1:nb
    figure('Position', [1000 100 500 550]);
    tiledlayout(4,3,'TileSpacing','tight','Padding','compact');
    for jj = 1:12
        nexttile
        ci = stats.equivalence_mdc{plt_order_r(jj), 1}.ci;
        lo = min(ci, [], 1);  hi = max(ci, [], 1);
        x = 0:100;
        hold on
        for cc = 1:3
            idx = sens.class{jj, bb} == cc;
            if any(idx)
                xs = x;  l = lo;  h = hi;
                xs(~idx) = NaN;  l(~idx) = NaN;  h(~idx) = NaN;
                plot(xs, l, '-', 'Color', cols(cc,:), 'LineWidth', 1.5)
                plot(xs, h, '-', 'Color', cols(cc,:), 'LineWidth', 1.5)
            end
        end
        yline( bounds(jj,bb), 'k--', 'LineWidth', 1)
        yline(-bounds(jj,bb), 'k--', 'LineWidth', 1)
        title(plttitles{jj})
        xlim([0 100]); xticks(0:20:100); xtickangle(0)
        if jj >= 10, xlabel('Gait Cycle (%)'); end
        if mod(jj,3) == 1, ylabel('Angle (°)'); end
        text(0.98, 0.98, sprintf('%d/%d/%d%%', round(sens.pct(jj,bb,1)), ...
            round(sens.pct(jj,bb,2)), round(sens.pct(jj,bb,3))), ...
            'Units','normalized','HorizontalAlignment','right', ...
            'VerticalAlignment','top','FontSize',7)
    end
    sgtitle(sprintf('Bounds: %s  (green = equivalent, red = not equivalent, grey = inconclusive; E/N/I %%)', bound_names{bb}), 'FontSize', 8)
end
