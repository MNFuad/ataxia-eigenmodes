% Demo script for a structural MBM analysis using cortical thickness maps.
%
% This is a GitHub-ready version of the original FA demo. It:
%   - uses only **relative paths** (no hard-coded /home/... paths)
%   - assumes input data live under `repo_root/data/mbm_demo_FA`
%   - configures the MBM structure and calls `mbm_main`
%   - saves results and an example plot of subject-level mode weights
%
% Expected directory layout (you can change `dataDir` below if needed):
%
%   repo_root/
%     matlab/
%       mbm/
%         01_structural_mbm_FA_demo.m
%     data/
%       mbm_demo_FA/
%         inputMaps_full_path.txt              % list of cortical maps
%         fsaverage_164k_cortex-lh_mask.txt    % cortical mask
%         G_two_sample.txt                     % design matrix (two-sample)
%         fsaverage_164k_midthickness-lh.vtk   % surface for plotting
%
% Requirements:
%   - MBM toolbox on the MATLAB path (mbm_main, beta_plot, etc.) 
%     available at https://github.com/NSBLab/MBM
%   - Statistics and Machine Learning Toolbox
%
% To run:
%   - adjust `dataDir` below if your layout differs
%   - ensure MBM functions are on the path
%   - run this script from MATLAB
%
% NOTE: This script is intentionally minimal and intended as a template.
%       You can modify file names, statistical test, number of eigenmodes,
%       etc. to match your own dataset.


clearvars
close all

% -------------------------------------------------------------------------
% Paths
% -------------------------------------------------------------------------
% Infer repository root assuming this file is under `matlab/mbm/`.
thisFile  = mfilename('fullpath');
scriptDir = fileparts(thisFile);
repoRoot  = fileparts(fileparts(scriptDir));  % matlab/mbm -> repo root

% Directory containing all input data for this demo. Adjust if needed.
dataDir = fullfile(repoRoot, 'data', 'mbm_example');

if ~isfolder(dataDir)
    warning('Data directory %s does not exist yet. Please create it and add the required files.', dataDir);
end

% ------------------------------------------------------------------------
%   add MBM toolbox to path
%  -------------------------------------------------------------------------
addpath(genpath(fullfile(repoRoot, 'matlab', 'tools', 'MBM-main')));

% -------------------------------------------------------------------------
% Random seed for reproducibility
% -------------------------------------------------------------------------
rng(2);  % fixed seed to make permutations reproducible

% -------------------------------------------------------------------------
% Input maps
% -------------------------------------------------------------------------
% You can either:
%   (a) use a 4D .mgh / .mat file
%   (b) provide a text file listing the full paths to each subject map
%
% Here we assume option (b): a list of file paths in `inputMaps_full_path.txt`.
MBM.maps.anatListFile = fullfile(dataDir, 'inputMaps_full_path.txt');
% Example alternatives:
% MBM.maps.anatListFile = fullfile(dataDir, 'inputMaps_ANCOVA_twosample.mgh');
% MBM.maps.anatListFile = fullfile(dataDir, 'inputMaps_ANCOVA_twosample.mat');

% Mask file (binary cortex mask with 1 = include, 0 = exclude)
MBM.maps.maskFile = fullfile(dataDir, 'fsaverage_164k_cortex-lh_mask.txt');

% -------------------------------------------------------------------------
% Statistical model
% -------------------------------------------------------------------------
% Choose one of the statistical tests by editing the lines below.
%
% Two-sample test (e.g. FA vs controls, no covariates):
MBM.stat.test       = 'two sample';
MBM.stat.designFile = fullfile(dataDir, 'G_two_sample.txt');
%
% Alternative examples (commented):
% MBM.stat.test       = 'ANCOVA';
% MBM.stat.designFile = fullfile(dataDir, 'G_ANCOVA.txt');
% MBM.stat.test       = 'one way ANOVA';
% MBM.stat.designFile = fullfile(dataDir, 'G_one_way_ANOVA.txt');

MBM.stat.nPer  = 5000;   % number of permutations
MBM.stat.pThr  = 0.1;    % tail estimation threshold
MBM.stat.thres = 0.05;   % significance threshold
MBM.stat.fdr   = false;  % FDR correction (set true if desired)

% -------------------------------------------------------------------------
% Eigenmodes
% -------------------------------------------------------------------------
% Option 1: let MBM compute eigenmodes from the .vtk surface.
% Option 2: supply precomputed eigenmodes and mass matrix.
%
% Here we only specify the number of eigenmodes to use; MBM will handle
% the rest. If you have precomputed modes, you can set:
%
% MBM.eig.eigFile  = fullfile(dataDir, 'fsaverage_164k_midthickness-lh_emode_200.mat');
% MBM.eig.massFile = fullfile(dataDir, 'fsaverage_164k_midthickness-lh_mass_200.mat');
%
% MBM.eig.nEigenmode = 200; % for real data in the paper, we used 200
MBM.eig.nEigenmode = 30;   % custom value , but must be <= 100 for the synthetic grid

% Save MBM result structure to a .mat file
MBM.eig.saveResult = true;
MBM.eig.resultFile = fullfile(dataDir, 'mbm_emp.mat');

% -------------------------------------------------------------------------
% Plotting configuration
% -------------------------------------------------------------------------
MBM.plot.visualize        = true;  % show figures
MBM.plot.saveFig          = false;  % save figure to .fig
MBM.plot.figFile          = fullfile(dataDir, 'demo_emp.fig');
MBM.plot.vtkFile          = fullfile(dataDir, 'fsaverage_164k_midthickness-lh.vtk');
MBM.plot.hemis            = 'left';   % hemisphere to analyse
MBM.plot.nInfluentialMode = 5;        % number of most influential modes to plot

% -------------------------------------------------------------------------
% Run MBM
% -------------------------------------------------------------------------
fprintf('Running MBM demo using data in: %s\n', dataDir);
MBM = mbm_main(MBM);

% Optionally save a PNG of the last figure:
% saveas(gcf, fullfile(dataDir, 'demo_emp_FA.png'));

% -------------------------------------------------------------------------
% Plot subject-wise beta weights (optional)
% -------------------------------------------------------------------------
if MBM.plot.saveFig  == 1
    subjectWeightDir = fullfile(dataDir, 'subject_weight', '\');
    if ~isfolder(subjectWeightDir)
        mkdir(subjectWeightDir);
    end
    
    % MBM.eig.beta_subject typically has size (nSubjects x nModes).
    % Here we skip the first column (often intercept / mean) and plot the rest.
    if isfield(MBM, 'eig') && isfield(MBM.eig, 'beta_subject')
        close all
        beta_plot(MBM.eig.beta_subject(:, 2:end), subjectWeightDir);
    else
        warning('MBM.eig.beta_subject not found – skipping beta_plot.');
    end
end
