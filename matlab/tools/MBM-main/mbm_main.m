function MBM = mbm_main(MBM)
% mbm_main is the main function of MBM toolbox which performs MBM analysis
%
%% Input:
% MBM       - Structure
%             MBM.maps    - Structure containing maps. Input fields are:
%                           MBM.maps.anatListFile - Character vector.
%                                                 - Path to either:
%                                                   + a text file comprising the list
%                                                 of paths to the anatomical maps
%                                                 in GIFTI, NIFTI, or .mgh format.
%                                                   + a .mat file
%                                                   containing a matrix
%                                                   whose each row is a map.
%                                                   + a .mgh file
%                                                   containing a 4-D matrix
%                                                   obtained as mri_glmgit
%                                                   output.
%
%                           MBM.maps.maskFile     - Character vector.
%                                                 - Path to a text file containing
%                                                   a binary mask where values '1' or
%                                                   '0' indicate the vertices of the
%                                                   applied maps to be used or removed.
%
%
%             MBM.stat    - Structure of parameters to produce a statistical map from
%                           the input maps for MBM analysis. Input fields are:
%                           MBM.stat.test         - Statistical test to be used:
%                                                 'one sample' one-sample t-test,
%                                                 'two sample' two-sample t-test,
%                                                 'one way ANOVA' one-way ANOVA,
%                                                 'ANCOVA' ANCOVA with two groups (f-test).
%
%                           MBM.stat.designFile    - Character vector.
%                                                  - Path to a text file containing a
%                                                    design matrix [m subjects by k effects].
%                                                  - For the design matrix in the statistical test:
%                                                           'one sample': one column, '1' or '0' indicates a subject in the group or not.
%                                                           'two sample': two columns, '1' or '0' indicates a subject in a group or not.
%                                                           'one way ANOVA': k columns, '1' or '0' indicates a subject in a group or not, number of subjects in each group must be equal.
%                                                           'ANCOVA': first column: '1' or another number (e.g., '2'): group effect (similar to input file for mri_glmfit in freesurfer)
%                                                                     second to k-th columns: covariates (discrete or continous numbers)
%
%                           MBM.stat.nPer         - Number
%                                                 - Number of permutations in the
%                                                   statistical test.
%
%                           MBM.stat.pThr         - Number
%                                                 - Threshold of p-values for
%                                                   tail approximation. If the
%                                                   p-values are below MBM.stat.pThr,
%                                                   these are refined further using a
%                                                   tail approximation from the
%                                                   Generalise Pareto Distribution (GPD).
%
%                           MBM.stat.thres        - Number
%                                                 - Threshold of p-values for
%                                                   being significant. When the
%                                                   p-value is below MBM.stat.thres,
%                                                   the statitical test is considered
%                                                   significant.
%
%                           MBM.stat.fdr          - Option ('true' or 'false') to
%                                                   correct multiple test with FDR or not.
%
%             MBM.eig     - Structure of MBM variables. Input fields are:
%                           MBM.eig.eigFile       - Character vector.
%                                                 - Path to a text file or .mat file containing
%                                                   eigenmodes in columns.
%                                                   If you do not have
%                                                   precalculated
%                                                   eigenmodes, this input
%                                                   can be omitted but
%                                                   MBM.plot.vtkFile is
%                                                   required. The code will
%                                                   calculate the
%                                                   eigenmodes from the
%                                                   surface given in
%                                                   MBM.plot.vtkFile.
%
%                           MBM.eig.massFile      - Character vector.
%                                                 - Path to a text file or .mat file containing
%                                                   the mass matrix when
%                                                   calculating the
%                                                   eigenmodes. See
%                                                   https://github.com/Deep-MI/LaPy
%                                                   and the reference
%                                                   papers to understand
%                                                   what is a mass matrix
%                                                   and how eigenmodes are
%                                                   calculated. If you do not have
%                                                   precalculated
%                                                   eigenmodes, this input
%                                                   can be omitted but
%                                                   MBM.plot.vtkFile is
%                                                   required. The code will
%                                                   calculate the mass
%                                                   matrix and
%                                                   eigenmodes from the
%                                                   surface given in
%                                                   MBM.plot.vtkFile.
%
%                           MBM.eig.nEigenmode    - Number
%                                                 - Number of eigenmodes to be used.
%
%
%                           MBM.eig.saveResult    - Option ('true' or 'false') to
%                                                   save the results, i.e., MBM
%                                                   structure.
%
%                           MBM.eig.resultFile    - Character vector.
%                                                 - Filename including path to save
%                                                   results.
%
%             MBM.plot    - Structure of parameters for plotting. Input fields are:
%
%                           MBM.plot.vtkFile      - Character vector.
%                                                 - Path to a vtk file containing a
%                                                   surface used to calculate the eigemodes and/or plot the results.
%
%                           MBM.plot.visualize    - Option ('true' or 'false') to
%                                                   visualize the results.
%
%                           MBM.plot.saveFig      - Option ('true' or 'false') to
%                                                   save the visualisation of the results.
%
%                           MBM.plot.figFile      - Character vector.
%                                                 - Filename including path to
%                                                   save the visualisation of the results.
%                                                 - File formats supported
%                                                 by 'saveas' in Matlab
%                                                 such as .fig .png .jpg
%                                                 .eps...
%
%                           MBM.plot.hemis        - 'left' or 'right' to visialise left or
%                                                   right hemisphere.
%
%                           MBM.plot.nInfluentialMode    - Number
%                                                        - Number of the most influential
%                                                          modes to plot.
%
%% Output:
% MBM       - Structure contains the following output fields:
%
%             MBM.maps    - MBM.maps.mask         - Vector of the binary mask.
%
%             MBM.stat    - Structure of parameters to produce a statistical map from
%                           the input maps for MBM analysis. Output fields are:
%
%                           MBM.stat.designMatrix  - Design matrix [m subjects by k effects].
%                                                  - For the design matrix in the statistical test:
%                                                           'one sample': one column, '1' or '0' indicates a subject in the group or not.
%                                                           'two sample': two columns, '1' or '0' indicates a subject in a group or not.
%                                                           'one way ANOVA': k columns, '1' or '0' indicates a subject in a group or not, number of subjects in each group must be equal.
%                                                           'ANCOVA': first column: '1' or another number (e.g., '2'): group effect (similar to input file for mri_glmfit in freesurfer)
%                                                                     second to k-th columns: covariates (discrete or continous numbers)
%
%
%                           MBM.stat.statMap      - Vector of a statistical map.
%
%                           MBM.stat.pMap         - Vector of p-values of the
%                                                   statistical map.
%
%                           MBM.stat.revMap       - Vector of "false" or "true"
%                                                   indicating the observed value of an
%                                                   element in the statistical map on
%                                                   the right or left tail of the null
%                                                   distribution.
%
%                           MBM.stat.thresMap     - Vector of a thresholded map.
%
%             MBM.eig     - Structure of MBM variables. Output fields are:
%                           MBM.eig.beta          - Vector of beta spectrum.
%
%                           MBM.eig.pBeta         - Vector of p-values of the
%                                                   beta spectrum.
%
%                           MBM.eig.revBeta       - Vector of "false" or "true"
%                                                   indicating the observed value of an
%                                                   element in the beta spectrum on
%                                                   the right or left tail of the null
%                                                   distribution.
%
%                           MBM.eig.significantBeta      - Vector of significant betas.
%
%                           MBM.eig.eig           - Matrix of columns of eigenmodes.
%
%                           MBM.eig.mass          - Mass matrix.
%
%                           MBM.eig.reconMap      - Vector of significant pattern.
%
%                           MBM.eig.betaOrder     - Vector of influential order.

% Trang Cao, Neural Systems and Behaviour Lab, Monash University, 2024.

%% initialisation
% addpath to the included packages or modify the path to the packages in your system
currentPath = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(currentPath,'func')))
addpath(fullfile(currentPath,'utils'))
addpath(fullfile(currentPath,'utils','modes'))
addpath(fullfile(currentPath,'utils','fdr_bh'))
addpath(fullfile(currentPath,'utils','PALM-master'))
addpath(fullfile(currentPath,'utils','gifti-matlab'))


% check input
mbm_check_input(MBM);

% read inputs from paths
[inputMap, MBM] = mbm_read_inputs(MBM);
mbm_check_read_inputs(MBM, inputMap);

% remove the unused vertices, e.g., the medial wall
inputMap = inputMap(:, MBM.maps.mask == 1);
MBM.eig.eig = MBM.eig.eig(MBM.maps.mask == 1, 1:MBM.eig.nEigenmode);
MBM.eig.mass = MBM.eig.mass(MBM.maps.mask == 1, MBM.maps.mask == 1);

%% SBM

% calculate statistical map
MBM.stat.statMap = mbm_stat_map(inputMap, MBM.stat);

% permutation tests on the statitical map
[statMapNull, MBM.stat] = mbm_perm_test_map(inputMap, MBM.stat);

% thresholded map
MBM.stat.thresMap = sign(MBM.stat.statMap);
MBM.stat.thresMap(MBM.stat.pMap > MBM.stat.thres) = 0;

%% MBM
% normalize the eigenmodes
MBM.eig.eig = mbm_normalize_eig(MBM.eig.eig, MBM.eig.nEigenmode);

%% === Yeo 7-network summary for each eigenmode (added) ===
% You need to set these two fields before calling mbm_main:
%   MBM.yeo7.annotFile : path to lh/rh Yeo2011_7Networks_N1000.annot
%   MBM.yeo7.useAbs    : 1 (default) use mean(abs(mode)), 0 use mean(mode)
%
% Example:
% MBM.yeo7.annotFile = '/path/to/lh.Yeo2011_7Networks_N1000.annot';
% MBM.yeo7.useAbs = 1;
if isfield(MBM, 'yeo7') && isfield(MBM.yeo7, 'annotFile') && ~isempty(MBM.yeo7.annotFile)

    if ~isfield(MBM.yeo7, 'useAbs'); MBM.yeo7.useAbs = 1; end

    % --- read Yeo7 labels from .annot ---
    % requires FreeSurfer MATLAB function read_annotation()
    [vertices, yeoLabelFull, ct] = read_annotation(MBM.yeo7.annotFile); %#ok<ASGLU>
    for i = 1:numel(ct.struct_names)
        switch ct.struct_names{i}
            case '7Networks_1'
                ct.struct_names{i} = 'Visual';
    
            case '7Networks_2'
                ct.struct_names{i} = 'Somatomotor';
    
            case '7Networks_3'
                ct.struct_names{i} = 'Dorsal Attention';
    
            case '7Networks_4'
                ct.struct_names{i} = 'Ventral Attention / Salience';
    
            case '7Networks_5'
                ct.struct_names{i} = 'Limbic';
    
            case '7Networks_6'
                ct.struct_names{i} = 'Frontoparietal Control';
    
            case '7Networks_7'
                ct.struct_names{i} = 'Default Mode';
    
            otherwise
                % ?? FreeSurfer_Defined_Medial_Wall
                % ?????
         end
     end

    % Apply the same mask used for maps/eigenmodes
    yeoLabel = yeoLabelFull(MBM.maps.mask == 1);

    % Unique network IDs present after masking
    netIDs = unique(yeoLabel(:))';
    % Remove unknown/unlabeled labels when present. In many .annot files this is 0,
    % while the medial wall can also appear as a named label in the colortable.
    netIDs(netIDs == 0) = [];
    nNet = numel(netIDs);

    % ---- build labelID -> name mapping (robust) ----
    % In .annot, yeoLabel contains integer IDs = ct.table(:,5), NOT 1..n
    hasCT = isstruct(ct) && isfield(ct, 'table') && isfield(ct, 'struct_names') ...
            && ~isempty(ct.table) && ~isempty(ct.struct_names);

    id2name = containers.Map('KeyType', 'double', 'ValueType', 'char');
    if hasCT
        % ct.table(:,5) are the encoded integer IDs used in "label"
        for r = 1:size(ct.table,1)
            id2name(double(ct.table(r,5))) = ct.struct_names{r};
        end
    end

    % Fallback names (common Yeo7 order) if colortable missing
    fallbackNames = {'Visual','Somatomotor','Dorsal Attention','Ventral Attention / Salience', ...
                     'Limbic','Frontoparietal Control','Default Mode'};

    % Determine netNames aligned to netIDs (one name per netID)
    netNames = cell(1, nNet);
    for j = 1:nNet
        thisID = double(netIDs(j));
        if hasCT && isKey(id2name, thisID)
            netNames{j} = id2name(thisID);
        else
            % if no colortable, we can only provide generic labels
            if j <= numel(fallbackNames)
                netNames{j} = fallbackNames{j};
            else
                netNames{j} = sprintf('label_%d', thisID);
            end
        end
    end

    % ---- compute network means for each eigenmode ----
    nMode = MBM.eig.nEigenmode;
    mean_abs    = nan(nMode, nNet);
    mean_signed = nan(nMode, nNet);

    for k = 1:nMode
        modeMap = MBM.eig.eig(:, k);  % already masked vertices
        for j = 1:nNet
            idx = (yeoLabel == netIDs(j));
            mean_signed(k, j) = mean(modeMap(idx), 'omitnan');
            mean_abs(k, j)    = mean(abs(modeMap(idx)), 'omitnan');
        end
    end

    % ---- choose metric based on useAbs (now it actually affects results) ----
    % useAbs = 1: mean(|mode|) within network (recommended)
    % useAbs = 0: |mean(mode)| within network (keeps "bias" strength, avoids sign flip issues)
    metric_all = nan(nMode, nNet);
    if MBM.yeo7.useAbs
        metric_all = mean_abs;
    else
        metric_all = abs(mean_signed);
    end

    % ---- top-2 networks per mode (by chosen metric) ----
    top2_idx   = zeros(nMode, 2);
    top2_name  = cell(nMode, 2);
    top2_value = nan(nMode, 2);

    for k = 1:nMode
        [~, order] = sort(metric_all(k, :), 'descend');
        nTop = min(2, numel(order));
        top2_idx(k, 1:nTop) = order(1:nTop);
        if nTop < 2
            top2_idx(k, 2) = order(1);
        end

        j1 = top2_idx(k, 1);
        j2 = top2_idx(k, 2);

        top2_name{k, 1}  = netNames{j1};
        top2_name{k, 2}  = netNames{j2};
        top2_value(k, 1) = metric_all(k, j1);
        top2_value(k, 2) = metric_all(k, j2);
    end

    % ---- Save into MBM ----
    MBM.eig.yeo7.netIDs        = netIDs;        % label IDs (encoded integers)
    MBM.eig.yeo7.netNames      = netNames;      % aligned to netIDs
    MBM.eig.yeo7.mean_abs      = mean_abs;
    MBM.eig.yeo7.mean_signed   = mean_signed;
    MBM.eig.yeo7.metric        = metric_all;    % what you used for ranking
    MBM.eig.yeo7.useAbs        = MBM.yeo7.useAbs;
    MBM.eig.yeo7.top2_idx      = top2_idx;      % indices into netIDs/netNames
    MBM.eig.yeo7.top2_name     = top2_name;
    MBM.eig.yeo7.top2_value    = top2_value;

    % ---- Optional: write a CSV summary ----
    if ~isfield(MBM.yeo7, 'csvFile') || isempty(MBM.yeo7.csvFile)
        if isfield(MBM.eig, 'resultFile')
            [p, n, ~] = fileparts(MBM.eig.resultFile);
            MBM.yeo7.csvFile = fullfile(p, [n '_yeo7_mode_summary.csv']);
        else
            MBM.yeo7.csvFile = fullfile(pwd, 'yeo7_mode_summary.csv');
        end
    end

    fid = fopen(MBM.yeo7.csvFile, 'w');
    if MBM.yeo7.useAbs
        fprintf(fid, 'mode,top1,top1_meanAbs,top2,top2_meanAbs\n');
    else
        fprintf(fid, 'mode,top1,top1_absMeanSigned,top2,top2_absMeanSigned\n');
    end

    for k = 1:nMode
        fprintf(fid, '%d,%s,%.6g,%s,%.6g\n', ...
            k, top2_name{k,1}, top2_value(k,1), top2_name{k,2}, top2_value(k,2));
    end
    fclose(fid);

end
%% === end Yeo7 summary ===

% eigenmode decomposision
% MBM.eig.beta = mbm_eigen_decompose(MBM.stat.statMap, MBM.eig.eig);
MBM.eig.beta = calc_eigendecomposition(MBM.stat.statMap', MBM.eig.eig, 'orthogonal', MBM.eig.mass);
MBM.eig.beta =  MBM.eig.beta';
beta_subject = zeros([size(inputMap,1) MBM.eig.nEigenmode]);
for i = 1: size(inputMap,1)
   beta_subject_individual = calc_eigendecomposition(inputMap(i,:)', MBM.eig.eig, 'orthogonal', MBM.eig.mass);
   beta_subject_indiviual = beta_subject_individual;
   beta_subject(i,:) = beta_subject_indiviual;
end
MBM.eig.beta_subject = beta_subject;
% permutation tests on the beta spectrum

MBM = mbm_perm_test_beta(statMapNull, MBM);

% significant betas
MBM.eig.significantBeta = MBM.eig.beta;
MBM.eig.significantBeta(MBM.eig.pBeta > MBM.stat.thres) = 0;

% sort significant beta
[betaSorted, MBM.eig.betaOrder] = sort(abs(MBM.eig.significantBeta), 'descend');

% sigificant pattern
MBM.eig.reconMap = MBM.eig.significantBeta * MBM.eig.eig';

%% plotting
if MBM.plot.visualize == 1
    mbm_plot(MBM);

end

%% saving results
if MBM.eig.saveResult == 1
    save(MBM.eig.resultFile, 'MBM');
end

end
