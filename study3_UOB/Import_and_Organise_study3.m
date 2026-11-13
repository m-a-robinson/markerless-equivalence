
% Import data from Sylvia's  study 3
% [UOB vs OB]

% Each participant contains both markerless data for 23 L&R gait curves. 
% Therefore a 23x12 array is imported and stored for each particpant


% Data folder
%cd 'C:\Users\marob\OneDrive - Liverpool John Moores University\RESEARCH\Research\Student Projects\PhD Students\Sylvia Augustine\SA MR Shared\Study 3\Data\Text UOB'
cd 'C:\Users\marob\OneDrive - Liverpool John Moores University\RESEARCH\Research\Student Projects\PhD Students\Sylvia Augustine\SA MR Shared\Study 3\Data\Text WU'


%% Define input data filenames

filenames = {'LAXV.txt','LAYV.txt','LAZV.txt'...
    'LFPV.txt',...
    'LHXV.txt','LHYV.txt','LHZV.txt',...
    'LKXV.txt','LKYV.txt','LKZV.txt',...
    'PEXV.txt','PEYV.txt','PEZV.txt',...
    'RAXV.txt','RAYV.txt','RAZV.txt',...
    'RFPV.txt',...
    'RHXV.txt','RHYV.txt','RHZV.txt',...
    'RKXV.txt','RKYV.txt','RKZV.txt'};

% Define paths for all participants

% Define the base path
%basePath = 'C:\Users\marob\OneDrive - Liverpool John Moores University\RESEARCH\Research\Student Projects\PhD Students\Sylvia Augustine\SA MR Shared\Study 3\Data\Text UOB\';
basePath = 'C:\Users\marob\OneDrive - Liverpool John Moores University\RESEARCH\Research\Student Projects\PhD Students\Sylvia Augustine\SA MR Shared\Study 3\Data\Text WU\';

% Create the cell array with 'P005' to 'P028'
numbers = 5:29;
cellArray = arrayfun(@(x) sprintf('P%03d', x), numbers, 'UniformOutput', false)';

% Join the base path with each element in the cell array
fullPaths = strcat(basePath, cellArray, '\ALL'); % data is in the ALL folder within a subject folder

% Ensure fullPaths is a 24x1 cell array
fullPaths = fullPaths(:); % This ensures it is a column vector

% Display the full paths
disp(fullPaths);

% CHECK FOR STUDY 3 - delete participant 26 from fullPaths?
% Note subject 20 is missing in the UOB data

%% Load files for all subjects
for ii = 1:length(fullPaths) %all subj

    % Import loop
    cd(fullPaths{ii}) % select the relevat data path
    pathfile = strcat(fullPaths{ii},'\',filenames);

    kk = 1;
    while kk <= size(filenames,2) % loop 23 input files
        % Import
        temp_import = importdata(pathfile{1,kk},'\t');
        % Remove unnecessary header data
        temp_all_data = temp_import.data;
        temp_all_data = temp_all_data(:,2:size((temp_all_data),2)); % remove first 1 cols
                % Check with Sylivia if mean + SD are exported
        data.subj{ii,1}.rawdata{kk,1} = temp_all_data; % store in "data" variable
        clear temp*
        kk = kk + 1;
    end

    clear path*
end

% %% Store the data for study 2 - NP - cols 5 & 6 only
% 
% for ii = 1:length(fullPaths) %all subj
%     for jj = 1:length(data.subj{1,1}.rawdata)
%         s2data.subj{ii,1}.mb{jj,1} = data.subj{ii,1}.rawdata{jj,5}; % ML
%         s2data.subj{ii,1}.ml{jj,1} = data.subj{ii,1}.rawdata{jj,6}; % MB
%     end
% end
