% Main script to run MBAM eigen-coefficient extraction for a task fMRI dataset.
%
% This script:
%   - infers the repository root from its own location
%   - sets parameters such as task name, sequence direction, smoothing, etc.
%   - defines a dataset root (`folder_name`) under `data/mbam_example`
%   - calls the function `eig_coef_func_zuitian_v5` to perform the analysis
%
% Requirements:
%   - The function `eig_coef_func.m` must be on the MATLAB path
%     (ideally in the same folder as this script: matlab/mbam/).
%   - MBM / MBAM / CIFTI toolboxes must be on your MATLAB path.
%   - Your fMRIPrep + task fMRI data must be organised under
%       <repo_root>/data/mbam_example/
%     in the way expected by `eig_coef_func_zuitian_v5` (subject IDs, design
%     matrices, confounds, etc.).
%
% To run:
%   1. Adjust the parameters in the "User parameters" section below.
%   2. Make sure your data is placed under `data/mbam_example/` (or update the
%      folder_name path).
%   3. Run this script from MATLAB.
%
% The function will:
%   - compute eigenmode coefficients for each subject
%   - save per-subject results
%   - save a group-level MBAM structure `<folder_name>/MBAM_out/MBAM_<task>_... .mat`
% -------------------------------------------------------------------------
% Example usage
% -------------
% From the MATLAB command window:
%
%   >> cd <path_to_repo>/matlab/mbam
%   >> 02_functional_mbam_task_eigcoefficients
%
% This will:
%   - infer <repo_root> from this file’s location
%   - set default parameters (task = 'ckcpass', nEigenmode = 200, etc.)
%   - use the dataset root:
%         <repo_root>/data/mbam_example/
%   - call:
%         MBAM = eig_coef_func(task, seq_dir, smoothsize, folder_name, IDfile, nEigenmode)
%
% Expected key files under `<repo_root>/data/mbam_example/`:
%   <repo_root>/data/mbam_example/subject_ids.txt     % list of subject IDs (one per line)
%   <repo_root>/data/mbam_example/atlas/...                        % atlas / masks / eigenmodes
%   <repo_root>/data/mbam_example/derivatives_prep/...          % subject-level fMRIPrep + FEAT outputs
%
% The function `eig_coef_func.m` must be on the MATLAB path
% (typically in the same folder as this script).
% -------------------------------------------------------------------------

clearvars
clc

%% ------------------------------------------------------------------------
%  Resolve repository root and data directory
%  -------------------------------------------------------------------------

% Get full path of this script
thisFile  = mfilename('fullpath');
scriptDir = fileparts(thisFile);             % .../matlab/mbam
repoRoot  = fileparts(fileparts(scriptDir)); % go up twice -> repo root

% Dataset root for this example (edit as needed)
% Root of MBAM example data
dataDir   = fullfile(repoRoot, 'data', 'mbam_example');

if ~isfolder(dataDir)
    warning('Data folder "%s" does not exist yet. Please create it and add your data.', dataDir);
end

%% ------------------------------------------------------------------------
%  User parameters (EDIT THESE)
%  -------------------------------------------------------------------------

% Task name as used in your fMRI filenames / FEAT directory names
% e.g. 'LANGUAGE', 'MOTOR', 'ckcpass', etc.
task = 'ckcpass';

% Direction of sequence (kept for compatibility with original code)
% Often 'LR' or 'RL'; use whatever convention you need. It is only used
% in the output filename.
seq_dir = 'LR';

% Smoothing size in mm. In the function, this controls whether surface
% data are read from .func.gii (0) or a smoothed .dtseries.nii file.
smoothsize = 0;

% Number of eigenmodes to analyse (must match available eigenmodes)
% nEigenmode = 200; % for real data in the paper, we used 200
nEigenmode = 30;   % custom value , but must be <= 100 for the synthetic grid

% Name (WITHOUT .txt) of the subject ID list under:
%   <dataDir>/mbam_example/<IDfile>.txt
%
% The file should contain one subject ID per line (e.g. "sub-0001").
IDfile = 'subject_ids';

%% ------------------------------------------------------------------------
%  Optional: add MBM / MBAM / CIFTI toolboxes to path
%  -------------------------------------------------------------------------
% If you don't already have these in your MATLAB path, you can add them here.
% For example (uncomment and edit according to your setup):

addpath(genpath(fullfile(repoRoot, 'matlab', 'tools', 'MBM-main')));
addpath(genpath(fullfile(repoRoot, 'matlab', 'tools', 'cifti-matlab')));

% Make sure `gifti`, `calc_geometric_eigenmode`, etc. are available.

%% ------------------------------------------------------------------------
%  Run MBAM eigen-coefficient extraction
%  -------------------------------------------------------------------------

fprintf('Running MBAM eigen-coefficient extraction...\n');
fprintf('  Task        : %s\n', task);
fprintf('  Direction   : %s\n', seq_dir);
fprintf('  Smooth size : %d mm\n', smoothsize);
fprintf('  nEigenmode  : %d\n', nEigenmode);
fprintf('  Data folder : %s\n', dataDir);
fprintf('  ID list     : %s.txt\n\n', IDfile);

MBAM = eig_coef_func(task, seq_dir, smoothsize, dataDir, IDfile, nEigenmode);

fprintf('\nDone. MBAM structure saved to:\n  %s\n', MBAM.eig.resultFile);
