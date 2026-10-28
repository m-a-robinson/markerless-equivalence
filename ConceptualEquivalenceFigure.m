%% Conceptual MDC & Equivalence figure

% Load data

% Subplot 1
% Raw hip data is from "gait_database_240117_FINALDATA_2022_PeerJ.mat"
% gait.allsubj.grandmeans.r1sw and gait.allsubj.grandmeans.r2sw 

cd('C:\Users\marob\OneDrive - Liverpool John Moores University\RESEARCH\Research\Student Projects\PhD Students\Sylvia Augustine\SA MR Shared\STUDY 1 & 2 Reliability Analysis')
load('gait_database_240117_FINALDATA_2022_PeerJ.mat');
s1_hipx = gait.allsubj.grandmeans.r1sw{16,1};
s2_hipx = gait.allsubj.grandmeans.r2sw{16,1};
s1_hipx_mean = gait.allsubj.grandmeans.r1sw{16,2};
s2_hipx_mean = gait.allsubj.grandmeans.r2sw{16,2};

% Subplot 2
% SEM for sportswear
% gait.allsubj.sw_ise
sem_hipx =gait.allsubj.sw_ise{16,1};

clear gait table*

% Subplot 3
% MDC data
cd('C:\Users\marob\OneDrive - Liverpool John Moores University\RESEARCH\Research\Student Projects\PhD Students\Sylvia Augustine\SA MR Shared\Study 2\New Data 2025')
load('study2data_20260226_withMDC.mat');

mdc_hipx = gait_allsubj_sw_mdc{7,1};
mdc_hipx_mean = gait_allsubj_sw_mdc{7,2};

% Subplot 4
% S2 raw data
mb_hipx = s2data.grandmeans.mb{18,1};
ml_hipx = s2data.grandmeans.ml{18,1};
mb_hipx_mean = s2data.grandmeans.mb{18,2};
ml_hipx_mean = s2data.grandmeans.ml{18,2};

% Subplot 5
% 90% CI Equivalence data
eq_hipx = stats.equivalence_mdc{18,1};


% Subplot 6
% eq_hipx and mdc_hipx_mean


%% Plot 



% 2x3 tiled figure for hipx plots
f(1) = figure(1);
t = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
% turn off box for all current axes in this figure
ax = findall(f(1),'Type','axes');
set(ax, 'Box', 'off');

% Common x vector (if needed) or assume data are already y-vectors plotted against 0:100
x = 0:100;

% Subplot 1: s1_hipx in cyan and s1_hipx in magenta, with text label
nexttile
plot(x, s1_hipx, 'Color', [0.65 0.65 0.65], 'LineWidth', 1.25); hold on
plot(x, s2_hipx, 'Color', [0.3010, 0.7450, 0.9330], 'LineWidth', 1.25); 
set(gca, 'Box', 'off');
title('a) Inter-session reliability data')
text(0.5, 0.95, 'Data source: Augustine et al. 2025', 'Units', 'normalized', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 11);
xlim([0 100])
xlabel('Gait Cycle (%)')
ylim([-30 60])
ylabel('Angle (°)')

% Subplot 2: sem_hipx in black
nexttile
plot(x, sem_hipx, 'm', 'LineWidth', 1.25);
set(gca, 'Box', 'off');
title('b) Inter-session variation')
xlim([0 100])
xlabel('Gait Cycle (%)')
ylim([0 10])

% Subplot 3: mdc_hipx and horizontal dashed line at mdc_hipx_mean (green)
nexttile
plot(x, mdc_hipx,'g', 'LineWidth', 1.25); hold on
set(gca, 'Box', 'off');
yline(mdc_hipx_mean, '--g', 'LineWidth', 1.5);
title('c) Minimum Detectable Change')
xlim([0 100])
xlabel('Gait Cycle (%)')
ylim([0 10])
% add label near the right end of the dashed line
text(98, mdc_hipx_mean + 0.05, 'MDC mean', ...
    'Units', 'data', ...
    'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'bottom', ...
    'FontSize', 10, ...
    'Color', 'g'); 

% Subplot 4 (row 2 col 1): mb_hipx in blue, ml_hipx in red
nexttile
hMB = plot(x, mb_hipx, 'b', 'LineWidth', 1.25); hold on
hML = plot(x, ml_hipx, 'r', 'LineWidth', 1.25);
set(gca, 'Box', 'off');
title('d) Marker-based vs Markerless')
text(0.5, 0.95, 'Data source: this study', 'Units', 'normalized', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 11);
xlim([0 100])
xlabel('Gait Cycle (%)')
ylim([-30 60])
ylabel('Angle (°)')

% create horizontal legend and remove box
% lgd = legend([hMB, hML], {'Markerless', 'Marker-based'}, ...
             %'Orientation', 'horizontal', 'FontSize', 9);
lgd = legend([hMB(1), hML(2)], {'Markerless','Marker-based'});
lgd.Box = 'off';
% NOTE STILL NEED TO MANUALLY ADJUST LEGEND AT THE MOMENT


% Subplot 5: eq_hipx.plot() using spm1d plotting
nexttile
eq_hipx.plot(); % assumes eq_hipx is an spm1d result object with plot method
set(gca, 'Box', 'off');
title('e) Mean difference & 90% CI')
xlim([0 100])
xlabel('Gait Cycle (%)')
ylim([-10 10])

% Subplot 6: eq_hipx.plot() and + / - mdc_hipx_mean horizontal lines in green
% nexttile
% eq_hipx.plot();
% hold on
% 
% % dashed MDC bounds
% yhigh =  mdc_hipx_mean;
% ylow  = -mdc_hipx_mean;
% yline(+yhigh, '--g', 'LineWidth', 1.5);
% yline(+ylow,  '--g', 'LineWidth', 1.5);
% 
% title('f) Equivalence test with MDC bounds')
% xlim([0 100])
% xlabel('Gait Cycle (%)')
% ylim([-10 10])
% 
% % draw and update positions
% drawnow
% 
% % place a double-headed vertical arrow at x = 10 between the two green dashed lines
% ax = gca;
% fig = gcf;
% 
% % convert data coords (x=10, ylow, yhigh) to figure-normalized coords
% axpos = ax.Position;         % [x y w h] in figure normalized units
% xdata = 10;
% xnorm = axpos(1) + (xdata - ax.XLim(1)) / diff(ax.XLim) * axpos(3);
% ynorm_bot = axpos(2) + (ylow  - ax.YLim(1)) / diff(ax.YLim) * axpos(4);
% ynorm_top = axpos(2) + (yhigh - ax.YLim(1)) / diff(ax.YLim) * axpos(4);
% 
% % annotation doublearrow (figure normalized units)
% hda = annotation(fig, 'doublearrow', [xnorm xnorm], [ynorm_bot ynorm_top], ...
%     'Color', [0 0.6 0], 'LineWidth', 1.5, 'Head1Style', 'vback2', 'Head2Style', 'vback2');
% 
% % add label "Equivalence region" at data y = -4 (so it moves with axes)
% text(xdata+2, -4.5, 'Equivalence region', ...
%     'Units', 'data', ...
%     'HorizontalAlignment', 'left', ...
%     'VerticalAlignment', 'middle', ...
%     'Color', [0 0.6 0], ...
%     'FontSize', 9);

% Subplot 6: eq_hipx.plot() with correct CI handling, red patches, arrow and label
nexttile
eq_hipx.plot();
set(gca, 'Box', 'off');
hold on

% dashed MDC bounds
yM = mdc_hipx_mean;        % scalar MDC mean
yline(+yM, '--g', 'LineWidth', 1.5);
yline(-yM, '--g', 'LineWidth', 1.5);

title('f) Equivalence test with MDC bounds')
xlim([0 100])
xlabel('Gait Cycle (%)')
ylim([-10 10])

drawnow  % ensure plot rendered

% --- get CI and x (swap rows: upper is second row, lower is first row) ---
ci_upper = eq_hipx.ci(2,:);   % corrected: upper CI
ci_lower = eq_hipx.ci(1,:);   % corrected: lower CI

% use eq_hipx.x if present otherwise default 0:100
if isprop(eq_hipx,'x') && ~isempty(eq_hipx.x)
    x = eq_hipx.x(:).';
else
    x = 0:100;
end

assert(numel(x)==numel(ci_upper) && numel(ci_upper)==numel(ci_lower), ...
    'x and CI vectors must be same length');

ax = gca;
set(gcf,'Renderer','opengl'); % ensure FaceAlpha works

% masks for threshold crossings (compare correctly to +yM)
pos_mask = ci_upper > +yM;    % CI upper exceeds +MDC
neg_mask = ci_lower < -yM;    % CI lower falls below -MDC

% helper to find contiguous true segments
find_segments = @(mask) deal(find(diff([0 mask(:).' 0])==1), find(diff([0 mask(:).' 0])==-1)-1);
[pos_starts, pos_ends] = find_segments(pos_mask);
[neg_starts, neg_ends] = find_segments(neg_mask);

createdPatches = gobjects(0);

% Positive patches: between +yM and ci_upper
for k = 1:numel(pos_starts)
    idx = pos_starts(k):pos_ends(k);
    xs = x(idx);
    ys_top = ci_upper(idx);
    ys_bot = +yM * ones(size(xs));
    xv = [xs, fliplr(xs)];
    yv = [ys_bot, fliplr(ys_top)];
    p = patch('XData', xv, 'YData', yv, ...
              'FaceColor', [1 0 0], 'EdgeColor', 'none', ...
              'FaceAlpha', 0.5, 'Parent', ax, 'HitTest', 'off');
    createdPatches(end+1) = p; %#ok<SAGROW>
end

% Negative patches: between ci_lower and -yM
for k = 1:numel(neg_starts)
    idx = neg_starts(k):neg_ends(k);
    xs = x(idx);
    ys_top = -yM * ones(size(xs));
    ys_bot = ci_lower(idx);
    xv = [xs, fliplr(xs)];
    yv = [ys_bot, fliplr(ys_top)];
    p = patch('XData', xv, 'YData', yv, ...
              'FaceColor', [1 0 0], 'EdgeColor', 'none', ...
              'FaceAlpha', 0.5, 'Parent', ax, 'HitTest', 'off');
    createdPatches(end+1) = p; %#ok<SAGROW>
end

% bring patches to front so they are visible above lines/CI
if ~isempty(createdPatches)
    for p = createdPatches(:).'
        uistack(p, 'top');
    end
end


labelOffset = 0.03 * diff(ax.YLim);  % vertical offset = 3% of y-range

% Positive labels: above the highest point of each positive patch (ci_upper)
for k = 1:numel(pos_starts)
    idx = pos_starts(k):pos_ends(k);
    x_mid = mean(x(idx));
    y_top = max(ci_upper(idx));          % highest point of the patch
    text(x_mid, y_top + labelOffset, 'Not equivalent', ...
        'Parent', ax, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'Color', [1 0 0], ...
        'FontWeight', 'bold', ...
        'FontSize', 9);
end

% Negative labels: above the top of the negative patch (-yM)
% (place slightly above -yM so label sits just above the patch)
for k = 1:numel(neg_starts)
    idx = neg_starts(k):neg_ends(k);
    x_mid = mean(x(idx));
    y_bottom = min(ci_lower(idx));       % lowest point of the negative patch
    text(x_mid, y_bottom - labelOffset, 'Not equivalent', ...
        'Parent', ax, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top', ...
        'Color', [1 0 0], ...
        'FontWeight', 'bold', ...
        'FontSize', 9);
end

drawnow

% place a double-headed vertical arrow at x = 10 between the two green dashed lines
fig = gcf;
axpos = ax.Position;         % [x y w h] in figure normalized units
xdata = 10;
xnorm = axpos(1) + (xdata - ax.XLim(1)) / diff(ax.XLim) * axpos(3);
ynorm_bot = axpos(2) + ((-yM) - ax.YLim(1)) / diff(ax.YLim) * axpos(4);
ynorm_top = axpos(2) + (yM - ax.YLim(1)) / diff(ax.YLim) * axpos(4);

annotation(fig, 'doublearrow', [xnorm xnorm], [ynorm_bot ynorm_top], ...
    'Color', [0 0.6 0], 'LineWidth', 1.5, 'Head1Style', 'vback2', 'Head2Style', 'vback2');

% add label "Equivalence region" at data y = -4.5
text(xdata+2, -4.5, 'Equivalence region', ...
    'Units', 'data', ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'middle', ...
    'Color', [0 0.6 0], ...
    'FontSize', 9);

% add label near the right end of the dashed line
text(98, mdc_hipx_mean + 0.05, 'MDC mean', ...
    'Units', 'data', ...
    'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'bottom', ...
    'FontSize', 10, ...
    'Color', 'g'); 

% Optional: adjust overall appearance
t.TileSpacing = 'compact';
t.Padding = 'compact';

exportgraphics(f(1),'conceptual_fig.jpg','Resolution',300)
%close(f(1))