%% Handwritten Digit Classification using Neural Network
% Main script - run this to train and evaluate the model
% Dataset: MNIST (loaded via MATLAB's built-in digitDataset or custom)

clc; clear; close all;

fprintf('=== Handwritten Digit Classification ===\n\n');

%% Step 1: Load Dataset
fprintf('Loading dataset...\n');

% Load MNIST-style digit data (built into MATLAB)
digitDatasetPath = fullfile(toolboxdir('nnet'), 'nndemos', 'nndatasets', 'DigitDataset');

if exist(digitDatasetPath, 'dir')
    imds = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    fprintf('Dataset loaded: %d images\n', numel(imds.Files));
else
    % Fallback: use MATLAB's digitTrain4DArrayData
    fprintf('Using built-in digit dataset...\n');
    [XTrain, YTrain, XTest, YTest] = load_builtin_dataset();
end

%% Step 2: Split into Train/Test
if exist('imds', 'var')
    [imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomize');
    fprintf('Train: %d | Test: %d\n', numel(imdsTrain.Files), numel(imdsTest.Files));
end

%% Step 3: Train the Model
fprintf('\nTraining model...\n');
if exist('imds', 'var')
    net = train_model(imdsTrain);
else
    net = train_model_array(XTrain, YTrain);
end

%% Step 4: Evaluate
fprintf('\nEvaluating model...\n');
if exist('imds', 'var')
    evaluate_model(net, imdsTest);
else
    evaluate_model_array(net, XTest, YTest);
end

fprintf('\nDone! Model training and evaluation complete.\n');
