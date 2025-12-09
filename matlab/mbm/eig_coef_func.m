function MBAM = eig_coef_func(task, dir, smoothsize, dataDir, IDfile, nEigenmode)
% eig_coef_func
%   Extract eigenmode coefficients for task fMRI on a simple MBAM layout.
%
% Expected directory layout:
%
%   dataDir/
%     subject_ids.txt              % or <IDfile>.txt
%     atlas/fsaverage/164k/lh/
%       fsaverage_164k_cortex-lh_mask.txt
%       fsaverage_164k_midthickness-lh.vtk
%     derivatives/sub-XXX/ses-1/func/
%       sub-XXX_ses-1_task-<task>_hemi-L_space-fsaverage_bold.func.gii
%       sub-XXX_ses-1_task-<task>_desc-confounds_timeseries.tsv
%     MBAM/Results/sub-XXX/MNINonLinear/tfMRI_<task>_ses-1.feat/
%       design_mat.txt   (task regressors)
%       design_con.txt   (contrast vector(s))
%
% Inputs
%   task        e.g. 'ckcpass'
%   dir         a label string, e.g. 'ses-1' or 'LR' (only used in output name)
%   smoothsize  0 -> read .func.gii (other values reserved for future dtseries)
%   dataDir     root folder of mbam data (e.g. fullfile(repo_root,'data','mbam_example'))
%   IDfile      basename of subject list (default 'subject_ids')
%   nEigenmode  number of geometric eigenmodes
%
% Output
%   MBAM struct saved to:
%     dataDir/MBAM/MBAM_data/MBAM_<task>_<dir>_<nEigenmode>_164k_lh.mat

% -------------------------------------------------------------------------
% Config
% -------------------------------------------------------------------------
hemisphere = 'lh';
Hemisphere = 'L';
resolution = '164k';

if nargin < 4 || isempty(dataDir)
    dataDir = pwd;
end
if nargin < 5 || isempty(IDfile)
    IDfile = 'subject_ids';
end

if smoothsize == 0
    filename_ext = '.func.gii';
else
    filename_ext = ['_smooth_', num2str(smoothsize), '_mm.dtseries.nii'];
end

% -------------------------------------------------------------------------
% Subject list
% -------------------------------------------------------------------------
idFile = fullfile(dataDir, [IDfile, '.txt']);
subject = importdata(idFile);
if iscell(subject)
    subject = char(string(subject));
else
    subject = char(string(subject));
end

% -------------------------------------------------------------------------
% Atlas: mask + surface -> eigenmodes
% -------------------------------------------------------------------------
atlasDir = fullfile(dataDir, 'atlas', 'fsaverage', resolution, hemisphere);
mask_ori = readmatrix(fullfile( ...
    atlasDir, ['fsaverage_', resolution, '_cortex-', hemisphere, '_mask.txt']));

MBAM.plot.vtkFile = fullfile( ...
    atlasDir, ['fsaverage_', resolution, '_midthickness-', hemisphere, '.vtk']);

[vertices, faces] = read_vtk(MBAM.plot.vtkFile);
vertices = vertices';
faces    = faces';

[vertices, faces, rois, mask] = trimExcludedRois(vertices, faces, mask_ori);
MBAM.maps.mask = mask';   % row vector

surface = struct('vertices', vertices, 'faces', faces);
surface = calc_geometric_eigenmode(surface, nEigenmode);

MBAM.eig.eig(mask,:)     = surface.evecs;
MBAM.eig.mass(mask,mask) = surface.mass;

MBAM.eig.eig  = MBAM.eig.eig(mask == 1, 1:nEigenmode);
MBAM.eig.mass = MBAM.eig.mass(mask == 1, mask == 1);
MBAM.eig.eig  = mbm_normalize_eig(MBAM.eig.eig, nEigenmode);

% Prep containers for subject-level coefficients
MBAM.eig.betaCon_subject  = [];
MBAM.eig.gammaCon_subject = [];

% -------------------------------------------------------------------------
% Loop over subjects
% -------------------------------------------------------------------------
for i = 1:size(subject,1)

    subj_id = strtrim(subject(i,:));
    fprintf('Processing %s (%d/%d)\n', subj_id, i, size(subject,1));

    % ----------------- fMRI surface time series (.func.gii) -----------------
    funcDir  = fullfile(dataDir, 'derivatives', subj_id, 'ses-1', 'func');
    funcFile = fullfile(funcDir, ...
        [subj_id, '_ses-1_task-', task, '_hemi-', Hemisphere, ...
         '_space-fsaverage_bold', filename_ext]);

    mygii      = gifti(funcFile);
    hemiCortex = mygii.cdata;                % [nVert x nTime]
    hemiCortex = hemiCortex(mask == 1,:);    % apply trimmed mask

    [nVoxel, nTimePoint] = size(hemiCortex);

    % ----------------- Project onto eigenmodes -----------------
    MBAM.eig.beta = calc_eigendecomposition( ...
        hemiCortex, MBAM.eig.eig, 'orthogonal', MBAM.eig.mass);
    alpha = MBAM.eig.beta';                  % [nTime x nModes]

    % ----------------- Task regressors (design_mat.txt) -----------------
    featDir = fullfile(dataDir, 'MBAM', 'Results', subj_id, ...
        'MNINonLinear', ['tfMRI_', task, '_ses-1.feat']);

    regressor_task = dlmread(fullfile(featDir, 'design_mat.txt'));
    regressor_task = regressor_task(1:nTimePoint,:);  % [nTime x nTaskReg]

    % ----------------- Confounds (fMRIPrep confounds TSV) -----------------
    confounds_tsv_file = fullfile(funcDir, ...
        [subj_id, '_ses-1_task-', task, '_desc-confounds_timeseries.tsv']);
    confounds_table = readtable(confounds_tsv_file, ...
        'FileType','text','Delimiter','\t');

    motion_cols = {'trans_x','trans_y','trans_z','rot_x','rot_y','rot_z'};
    nMotion = numel(motion_cols);
    motion_params = zeros(nTimePoint, nMotion);
    for c = 1:nMotion
        if ismember(motion_cols{c}, confounds_table.Properties.VariableNames)
            temp_col = confounds_table.(motion_cols{c})(1:nTimePoint);
            motion_params(:,c) = fillmissing(temp_col, 'constant', 0);
        end
    end

    cosine_idx = find(contains(confounds_table.Properties.VariableNames,'cosine'));
    if ~isempty(cosine_idx)
        cosine_params = table2array(confounds_table(1:nTimePoint, cosine_idx));
    else
        cosine_params = [];
    end

    % ----------------- Build GLM design -----------------
    regressor = [regressor_task, motion_params, cosine_params];
    regressor = [ones(nTimePoint,1), regressor];   % add intercept

    % Contrast matrix (design_con.txt; flattened)
    conVec   = dlmread(fullfile(featDir, 'design_con.txt'));
    conMat   = reshape(conVec, [], size(regressor_task,2));    % [nCon x nTaskReg]
    nContrast = size(conMat,1);
    nCosine   = size(cosine_params,2);

    % Pad contrasts with zeros for motion+cosine+intercept
    conMat = [conMat, zeros(nContrast, nMotion + nCosine)];
    conMat = [zeros(nContrast,1), conMat];  % intercept

    % ----------------- GLM in voxel space (gamma) -----------------
    gamma    = (regressor.' * regressor) \ (regressor.' * hemiCortex');  % [nReg x nVoxel]
    gammaCon = conMat * gamma;                                           % [nCon x nVoxel]

    % ----------------- GLM in eigenmode space (beta) -----------------
    beta     = (regressor.' * regressor) \ (regressor.' * alpha);        % [nReg x nModes]
    betaCon  = conMat * beta;                                            % [nCon x nModes]

    % For demo: skip full t-stat computation; placeholders:
    tGamma    = zeros(size(gammaCon));
    tGammaCDF = zeros(size(gammaCon));
    tBeta     = zeros(size(betaCon));
    tBetaCDF  = zeros(size(betaCon));

    % Store first contrast across modes for this subject
    MBAM.eig.betaCon_subject(:,i)  = betaCon(1,:).';
    MBAM.eig.gammaCon_subject(:,i) = gammaCon(1,:).';

    % Save per-subject MBAM eigen-coeffs
    outFile = fullfile(featDir, ...
        ['beta_eig_', num2str(nEigenmode), '_', num2str(smoothsize), ...
         '_', resolution, '_', hemisphere, '.mat']);

    save(outFile, 'betaCon','gammaCon', ...
                  'tBeta','tGamma','tBetaCDF','tGammaCDF');
end

% -------------------------------------------------------------------------
% Save group-level MBAM struct
% -------------------------------------------------------------------------
saveRoot = fullfile(dataDir, 'MBAM', 'MBAM_data');
if ~exist(saveRoot, 'dir'); mkdir(saveRoot); end

MBAM.eig.resultFile = fullfile(saveRoot, ...
    ['MBAM_', task, '_', dir, '_', num2str(nEigenmode), '_', ...
     resolution, '_', hemisphere, '.mat']);

save(MBAM.eig.resultFile, 'MBAM');
fprintf('\nMBAM eigen-coefficients saved to:\n  %s\n', MBAM.eig.resultFile);

end
