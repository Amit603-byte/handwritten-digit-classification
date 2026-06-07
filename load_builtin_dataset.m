function [XTrain, YTrain, XTest, YTest] = load_builtin_dataset()
%LOAD_BUILTIN_DATASET Load MATLAB's built-in handwritten digit dataset
%
%   Returns 4D arrays of digit images and corresponding labels.
%   Uses MATLAB's digitTrain4DArrayData and digitTest4DArrayData.
%
%   Output:
%       XTrain - [28 x 28 x 1 x N] training images
%       YTrain - [N x 1] training labels (categorical)
%       XTest  - [28 x 28 x 1 x M] test images
%       YTest  - [M x 1] test labels (categorical)

fprintf('Loading built-in MATLAB digit dataset...\n');

[XTrain, YTrain] = digitTrain4DArrayData();
[XTest,  YTest ] = digitTest4DArrayData();

fprintf('Training samples : %d\n', size(XTrain, 4));
fprintf('Test samples     : %d\n', size(XTest, 4));
fprintf('Classes          : %s\n', strjoin(string(categories(YTrain)), ', '));

end
