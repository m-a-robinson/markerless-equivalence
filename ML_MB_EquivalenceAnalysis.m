%% Run equivalence tests for new Study 2 data (Sept 2025)

% Step 1. Calculate condition/joint means
% Step 2. Plot all joints and components as a visual check
% Step 3. Run equivalence tests between ML and MB data
% Step 4. Plot equivalence test outcomes

%% Calculate means for each particpant / joint angle

% Stored in column two next to the trial data

for ii = 1:length(s2data.subj) % 24 participants
    for jj = 1:23 % angles
        s2data.subj{ii,1}.mb{jj,2} = mean(s2data.subj{ii,1}.mb{jj,1},2);
        s2data.subj{ii,1}.ml{jj,2} = mean(s2data.subj{ii,1}.ml{jj,1},2);
    end 
end

%% Calculate grand means and SDs
% Grandmeans are across all subjects

% All subject data are stored in col 1, grandmeans col 2, sds in col 3.

jj=1;
while jj < 24 % angles

    for ii = 1:length(s2data.subj) % each participant

        temp_mb_ang(:,ii) = s2data.subj{ii,1}.mb{jj,2};
        temp_ml_ang(:,ii) = s2data.subj{ii,1}.ml{jj,2};
        
    end

    s2data.grandmeans.mb{jj,1} = temp_mb_ang;
    s2data.grandmeans.mb{jj,2} = mean(temp_mb_ang, 2);
    s2data.grandmeans.mb{jj,3} = std(temp_mb_ang, 0, 2);

    s2data.grandmeans.ml{jj,1} = temp_ml_ang;
    s2data.grandmeans.ml{jj,2} = mean(temp_ml_ang, 2);
    s2data.grandmeans.ml{jj,3} = std(temp_ml_ang, 0, 2);
    
    jj = jj +1;
    clear temp*

end


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
ylimmax = [30, 15, 30, 60, 30, 30, 70, 30, 30, 30, 30, 30];
ylimmin = [-10, -15, -30, -30, -30, -30, -10, -30, -30, -30, -30, -30];
txtloc = ylimmax - ((ylimmax - ylimmin)*0.15); 
plt_order_l = [11,12,13,5,6,7,8,9,4,1,2,3]; % needed to switch the subplot order
plt_order_r = [11,12,13,18,19,20,21,22,17,14,15,16]; % needed to switch the subplot order

pltylabels2={'St. Dev. (deg)','','','St. Dev. (deg)','','', 'St. Dev. (deg)','','','St. Dev. (deg)','','',};
txtloc2 = ones(1,12) * (6 - ((6 - 0)*0.15)); 
txtloc3 = ones(1,12) * 0.8; 


%% ML MB data All subject means
% This plots individual subject data not grandmeans/SDs

% All participant means
f(1) = figure(1);
f(1).Position(1:4) = [1000 100 500 550];
for ii = 1:length(s2data.subj) % number subjects
    for jj = 1:12 % gait variables
        subplot(4,3,jj)
        sgtitle('Means MB (blue), ML (red), RIGHT')
        
        plot(s2data.subj{ii,1}.mb{plt_order_r(jj), 2},'b') % markerbased means
        hold on
        plot(s2data.subj{ii,1}.ml{plt_order_r(jj), 2},'r') % markerless
        
        yline(0)
        title(plttitles(1,jj))
        xticks([0, 20, 40, 60, 80, 100]);
        xtickangle(0)
        xlabel(pltxlabels(1,jj))
        ylabel(pltylabels(1,jj))
        xlim([0 101])
        ylim([ylimmin(jj),ylimmax(jj)])             
    end
end

exportgraphics(f(1),'mlmb_allmeans_jan2026.jpg','Resolution',300)
close(f(1))



%% ML MB data plot grand means (calculated in spm1d)

% Used for final means figure

f(2) = figure(2);
f(2).Position(1:4) = [1000 100 500 550];
t1 = tiledlayout(4,3,'TileSpacing','tight','Padding','compact');
% Mean plot with standard deviation
for jj = 1:12 % gait variables

        nexttile
        %sgtitle('Grand Means MB (blue), ML (red), RIGHT')
        spm1d.plot.plot_meanSD(s2data.grandmeans.mb{plt_order_r(jj), 1}','color','b');
        hold on
        spm1d.plot.plot_meanSD(s2data.grandmeans.ml{plt_order_r(jj), 1}','color','r');
        title(plttitles(1,jj))
        yline(0)
        xticks([0, 20, 40, 60, 80, 100]);
        xtickangle(0)
        xlabel(pltxlabels(1,jj))
        ylabel(pltylabels(1,jj))
        xlim([0 101])
        ylim([ylimmin(jj),ylimmax(jj)])
        box off

        %txt = ['RMSD: ' num2str(gait.allsubj.rmsd_cw(plt_order_r(jj),1),3)];
        %text(50,txtloc(jj),txt,'FontSize',8, 'HorizontalAlignment','center')

        if jj == 2
            % dummy handles for legend entries
            hMarkerless = plot(nan, nan, '-r', 'LineWidth', 2);
            hMarkerBased  = plot(nan, nan, '-b', 'LineWidth', 2);

            % create legend and get handle
            lgd = legend([hMarkerless, hMarkerBased], {'Markerless', 'Marker-based'}, ...
                'Location', 'northoutside', 'Orientation', 'horizontal', ...
                'FontSize', 9);

            % remove box
            lgd.Box = 'off';

            % attach legend to the tiledlayout top center
            % (places legend in the layout's north tile)
            lgd.Layout.Tile = 'north';
        end
end

%exportgraphics(f(2),'mlmb_mean_sds_apr2026.jpg','Resolution',300)
%close(f(2))


%% Equivalence plots


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

    stats.equivalence{jj,1} = spm1d.stats.ci_pairedsample(s2data.grandmeans.mb{jj,1}',...
        s2data.grandmeans.ml{jj, 1}',...
        alpha,...
        'difference', 0);
        
end 

%%%--------------------KEY PART--------------------------%%%
% Define upper and lower equivalence bound (arbitrary at this stage)
% should be an informed choice 
LEB = -5;
UEB = 5;
%%%------------------------------------------------------%%%


f(3) = figure(3);
f(3).Position(1:4) = [1000 100 500 550];
t1 = tiledlayout(4,3,'TileSpacing','tight','Padding','compact');
% Mean plot with standard deviation
for jj = 1:12 % gait variables

        nexttile
        sgtitle('Equivalence tests')
        stats.equivalence{plt_order_r(jj), 1}.plot()
        title(plttitles(1,jj))
        xlim([0, 100])
        ylabel('Angle (deg)')
        hold on
        yline(LEB,'g--','Linewidth',1.5)
        %text(0.1,LEB*1.1,'equivalence bound')
        yline(UEB,'g--','Linewidth',1.5)
        %text(0.1,UEB*1.1,'equivalence bound')
        xticks([0, 20, 40, 60, 80, 100]);
        xtickangle(0)
        xlabel(pltxlabels(1,jj))
        ylabel(pltylabels(1,jj))
        xlim([0 101])

end

%exportgraphics(f(3),'mlmb_equivalence_jan2026.jpg','Resolution',300)
%close(f(3))
