function [predictedLabel, confidence] = predict_digit(net, imagePath)
%PREDICT_DIGIT Classify a handwritten digit from an image file
%
%   [predictedLabel, confidence] = predict_digit(net, imagePath)
%
%   Input:
%       net       - Trained neural network (from train_model.m)
%       imagePath - Full path to the image file (.png, .jpg, .bmp)
%
%   Output:
%       predictedLabel - Predicted digit (0-9) as string/categorical
%       confidence     - Confidence score (0 to 1)
%
%   Example:
%       load('trained_digit_classifier.mat');
%       [label, conf] = predict_digit(net, 'my_digit.png');
%       fprintf('Predicted: %s (%.1f%%)\n', char(label), conf*100);

%% Load and preprocess the image
if ischar(imagePath) || isstring(imagePath)
    img = imread(imagePath);
else
    % Assume raw image matrix was passed
    img = imagePath;
end

imgProcessed = preprocess_image(img);

% Reshape to [28 28 1 1] for network input
imgInput = reshape(imgProcessed, [28, 28, 1, 1]);

%% Classify
[predictedLabel, scores] = classify(net, imgInput);
confidence = max(scores);

%% Display result
fprintf('\n--- Prediction Result ---\n');
fprintf('Predicted Digit : %s\n', char(predictedLabel));
fprintf('Confidence      : %.2f%%\n', confidence * 100);

%% Show the image with prediction
figure('Name', 'Digit Prediction', 'NumberTitle', 'off');
imshow(imgProcessed, []);
title(sprintf('Predicted: %s  |  Confidence: %.1f%%', ...
    char(predictedLabel), confidence * 100), ...
    'FontSize', 14, 'FontWeight', 'bold');
colormap(gray);

end
