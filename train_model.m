function net = train_model(imdsTrain)
%TRAIN_MODEL Train a CNN for handwritten digit classification
%   net = train_model(imdsTrain) trains a convolutional neural network
%   on the provided image datastore and returns the trained network.
%
%   Input:
%       imdsTrain - ImageDatastore with training images and labels
%   Output:
%       net       - Trained SeriesNetwork object

%% Define CNN Architecture
layers = [
    imageInputLayer([28 28 1], 'Name', 'input', 'Normalization', 'zerocenter')

    % Conv Block 1
    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv_1')
    batchNormalizationLayer('Name', 'bn_1')
    reluLayer('Name', 'relu_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool_1')

    % Conv Block 2
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv_2')
    batchNormalizationLayer('Name', 'bn_2')
    reluLayer('Name', 'relu_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool_2')

    % Conv Block 3
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_3')
    batchNormalizationLayer('Name', 'bn_3')
    reluLayer('Name', 'relu_3')

    % Fully Connected Layers
    fullyConnectedLayer(128, 'Name', 'fc_1')
    reluLayer('Name', 'relu_fc')
    dropoutLayer(0.5, 'Name', 'dropout')
    fullyConnectedLayer(10, 'Name', 'fc_output')

    % Output
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

%% Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate',     0.001, ...
    'MaxEpochs',            15, ...
    'MiniBatchSize',        64, ...
    'Shuffle',              'every-epoch', ...
    'ValidationFrequency',  30, ...
    'Verbose',              true, ...
    'Plots',                'training-progress', ...
    'LearnRateSchedule',    'piecewise', ...
    'LearnRateDropFactor',  0.1, ...
    'LearnRateDropPeriod',  10);

%% Train the Network
fprintf('Starting CNN training...\n');
net = trainNetwork(imdsTrain, layers, options);
fprintf('Training complete.\n');

%% Save the trained model
save('trained_digit_classifier.mat', 'net');
fprintf('Model saved to trained_digit_classifier.mat\n');

end


%% Helper: Train on raw 4D array (fallback)
function net = train_model_array(XTrain, YTrain)
%TRAIN_MODEL_ARRAY Fallback trainer using 4D array input

numClasses = numel(unique(YTrain));

layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(64)
    reluLayer
    dropoutLayer(0.4)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs',        10, ...
    'MiniBatchSize',    128, ...
    'Verbose',          true, ...
    'Plots',            'training-progress');

net = trainNetwork(XTrain, categorical(YTrain), layers, options);
save('trained_digit_classifier.mat', 'net');
end
