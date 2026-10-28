%% Run equivalence tests for new Study 2 data (Feb 2026)
% Using MDC not standard equivalence bounds


% Step 1. Calculate condition/joint means
% Step 2. Plot all joints and components as a visual check
% Step 3. Run equivalence tests between ML and MB data
% Step 4. Plot equivalence test outcomes


%% Define titles
% plttitles={'Pelvic Tilt','Pelvic Ob.','Pelvic Rot.',...
%     'Hip X','Hip Y','Hip Z',...
%     'Knee X','Knee Y','Knee Z',...
%     'Ankle X','Ankle Y','Ankle Z'};

plttitles = {'Pelvis Anterior Tilt', 'Pelvis Obliquity', 'Pelvis Rotation',...
    'Hip Flexion', 'Hip Adduction', 'Hip Rotation',...
    'Knee Flexion', 'Knee Varus', 'Foot Progression',...
    'Ankle Dorsiflexion', 'Ankle Inversion', 'Ankle Rotation'};


pltylabels={'Angle (°)','','','Angle (°)','','', 'Angle (°)','','','Angle (°)','','',};
pltxlabels={'','','','','','','','','','Gait Cycle (%)','Gait Cycle (%)','Gait Cycle (%)'};
ylimmax = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20];
ylimmin = [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -20, -20];
txtloc = ylimmax - ((ylimmax - ylimmin)*0.15); 
plt_order_l = [11,12,13,5,6,7,8,9,4,1,2,3]; % needed to switch the subplot order
plt_order_r = [11,12,13,18,19,20,21,22,17,14,15,16]; % needed to switch the subplot order

pltylabels2={'St. Dev. (deg)','','','St. Dev. (deg)','','', 'St. Dev. (deg)','','','St. Dev. (deg)','','',};
txtloc2 = ones(1,12) * (6 - ((6 - 0)*0.15)); 
txtloc3 = ones(1,12) * 0.8; 


%% Equivalence tests & plots


% Mean difference and CI
% CI - 90% 
% "A 90% confidence interval (CI; 1   2a) is used instead of a 95% CI
% (1   a) because two one-sided tests (each with an a of 5%) are
% performed."
alpha     = 0.1; % change from 0.05 for a 90% CI? 

% CI method - Dixon et al 2018
% The null hypothesis of non-equivalence is rejected at level α if the 
% 100(1 − 2α)% confidence interval for the difference in means lies
% entirely with the equivalence region.
for jj = 1:23 % gait variables

    stats.equivalence_mdc{jj,1} = spm1d.stats.ci_pairedsample(s2data.grandmeans.mb{jj,1}',...
        s2data.grandmeans.ml{jj, 1}',...
        alpha,...
        'difference', 0);
        
end 

%%%--------------------KEY PART--------------------------%%%
% Define upper and lower equivalence bound (Feb 2026 - using MDC)
% This uses the SW MDC data from study 1 from "gait_allsubj_sw_mdc"
% Use gait_allsubj_swcw_mdc if you want inter-session-clothing mdc

% LEB = -5;
% UEB = 5;

% In gait_allsubj_swcw_mdc the variable order is 
% Sagittal pelvis, frontal pelvis, transverse pelvis,
% Sagittal ankle, frontal ankle, transverse ankle,
% Sagittal hip, frontal hip, transverse hip,
% Sagittal knee, frontal knee, transverse knee,
% sag_order = [1, 7, 10, 4];%pel,hip,kne,ank SAGITTAL
% fro_order = [2, 8, 11, 5];%pel,hip,kne,ank FRONTAL
% tra_order = [3, 9, 12, 6];%pel,hip,kne,ank TRANSVERSE

% these need to be put in an order to match the study 2 data
% plt_order_r = [11,12,13,18,19,20,21,22,17,14,15,16]; % needed to switch the subplot order

% plttitles = {'Pelvis Anterior Tilt', 'Pelvis Obliquity', 'Pelvis Rotation',...
%     'Hip Flexion', 'Hip Adduction', 'Hip Rotation',...
%     'Knee Flexion', 'Knee Varus', 'Foot Progression',...
%     'Ankle Dorsiflexion', 'Ankle Inversion', 'Ankle Rotation'};

mdc_order = [1,2,3,7,8,9,10,11,12,4,5,6];
LEB = cell2mat(gait_allsubj_sw_mdc(mdc_order, 2))*-1;  
UEB = cell2mat(gait_allsubj_sw_mdc(mdc_order, 2));  

% replace Knee Z with -5 +5 as default values as study 1 didn't examine 
% foot progression.
LEB(9) = -5;
UEB(9) = 5;

% ---------------------------------------------------

% Identify when the equivalence thresholds are crossed. 
% Move everything into a new, aligned structure 

% {'Pelvis Anterior Tilt', 'Pelvis Obliquity', 'Pelvis Rotation',...
%     'Hip Flexion', 'Hip Adduction', 'Hip Rotation',...
%     'Knee Flexion', 'Knee Varus', 'Foot Progression',...
%     'Ankle Dorsiflexion', 'Ankle Inversion', 'Ankle Rotation'};

for jj = 1:12 % gait variables
    ci_data{jj,1} = stats.equivalence_mdc{plt_order_r(jj), 1}.ci;
    ci_data{jj,2} = gait_allsubj_sw_mdc{mdc_order(jj),2};
end

% replace Knee Z with -5 +5 as default values as study 1 didn't examine 
ci_data{9,2} = 5;

% Check if CI data is inside equivalence thresholds, calculate % of time
% within equivalence threshold. Store in 3rd column. 
% Also add range of percentage crossings - in 4th column

for jj = 1:12
    temp_mat   = ci_data{jj,1};    % 2x101
    temp_limit = ci_data{jj,2};    % scalar

    temp_col_within = all(abs(temp_mat) <= temp_limit, 1);   % 1x101 logical
    temp_pct_columns_within = 100 * sum(temp_col_within) / 101;
    ci_data{jj,3} = temp_pct_columns_within;

    idx = find(temp_col_within);
    if isempty(idx)
        ranges = zeros(0,2);
    else
        breaks = find(diff(idx) > 1);
        starts = idx([1, breaks+1]);
        ends   = idx([breaks, numel(idx)]);
        ranges = [starts(:), ends(:)];
    end
    ci_data{jj,4} = ranges;

    fprintf('Columns within ±%g: %d / 101 (%.2f%%)\n', temp_limit, sum(temp_col_within), temp_pct_columns_within);
end


%% PLOT
%%%------------------------------------------------------%%%

% Normal Equivalence plots with MDC
f(3) = figure(3);
f(3).Position(1:4) = [1000 100 500 550];
t1 = tiledlayout(4,3,'TileSpacing','tight','Padding','compact');
% Mean plot with standard deviation
for jj = 1:12 % gait variables

        nexttile
        %sgtitle('Equivalence tests')
        stats.equivalence_mdc{plt_order_r(jj), 1}.plot()
        title(plttitles(1,jj))
        xlim([0, 100])
        ylabel('Angle (deg)')
        hold on
        %yline(0,':k','LineWidth',1)  
        yline(LEB(jj),'g--','Linewidth',1.5)
        %text(0.1,LEB*1.1,'equivalence bound')
        yline(UEB(jj),'g--','Linewidth',1.5)
        %text(0.1,UEB*1.1,'equivalence bound')
        xticks([0, 20, 40, 60, 80, 100]);
        xtickangle(0)
        xlabel(pltxlabels(1,jj))
        ylabel(pltylabels(1,jj))
        ylim([ylimmin(jj),ylimmax(jj)+5])

        text(0.98, 0.98, sprintf('%d%% Equiv.', round(ci_data{jj,3})), ...
            'Units', 'normalized', ...
            'HorizontalAlignment', 'right', ...
            'VerticalAlignment', 'top', ...
            'FontSize', 8, ...
            'Color', [0,0.9,0.1]);

end

exportgraphics(f(3),'mlmb_equivalence_mdc_feb2026_percent.jpg','Resolution',300)
%close(f(3))


%% Normal Equivalence plots with with time-varying MDC thresholds 
f(4) = figure(4);
f(4).Position(1:4) = [1000 100 500 550];
t1 = tiledlayout(4,3,'TileSpacing','tight','Padding','compact');
% Mean plot with standard deviation
for jj = 1:12 % gait variables

        nexttile
        sgtitle('Equivalence tests')
        stats.equivalence_mdc{plt_order_r(jj), 1}.plot()
        title(plttitles(1,jj))
        xlim([0, 100])
        ylabel('Angle (deg)')
        hold on
        plot(gait_allsubj_sw_mdc{mdc_order(jj),1},'g--','Linewidth',1.5)
        plot(gait_allsubj_sw_mdc{mdc_order(jj),1}*-1,'g--','Linewidth',1.5)
        % yline(LEB(jj),'g--','Linewidth',1.5)
        %text(0.1,LEB*1.1,'equivalence bound')
        %yline(UEB(jj),'g--','Linewidth',1.5)
        %text(0.1,UEB*1.1,'equivalence bound')
        xticks([0, 20, 40, 60, 80, 100]);
        xtickangle(0)
        xlabel(pltxlabels(1,jj))
        ylabel(pltylabels(1,jj))
        xlim([0 101])



end

%exportgraphics(f(4),'mlmb_equivalence_mdc_101_feb2026.jpg','Resolution',300)
%close(f(3))

