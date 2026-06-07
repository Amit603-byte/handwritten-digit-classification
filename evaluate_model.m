function evaluate_model(net, imdsTest)
%EVALUATE_MODEL Evaluate trained network on test set
%
%   evaluate_model(net, imdsTest) classifies all images in the test
%   datastore, computes accuracy, and displays a confusion matrix.
%
%   Input:
%       net      - Trained neural network
%       imdsTest - ImageDatastore with test images and true labels

fprintf('\n=== Model Evaluation ===\n');

%% Classify test images
fprintf('Classifying test images...\n');
YPred = classify(net, imdsTest, 'MiniBatchSize', 64);
YTrue = imdsTest.Labels;

%% Overall accuracy
accuracy = sum(YPred == YTrue) / numel(YTrue) * 100;
fprintf('Test Accuracy: %.2f%%\n', accuracy);

%% Per-class accuracy
classes = categories(YTrue);
fprintf('\nPer-class accuracy:\n');
fprintf('%-10s %-10s %-10s %-10s\n', 'Digit', 'Correct', 'Total', 'Accuracy');
fprintf('%s\n', repmat('-', 1, 45));

for i = 1:numel(classes)
    classIdx = YTrue == classes{i};
    classAcc = sum(YPred(classIdx) == YTrue(classIdx)) / sum(classIdx) * 100;
    fprintf('%-10s %-10d %-10d %.2f%%\n', ...
        classes{i}, sum(YPred(classIdx) == YTrue(classIdx)), sum(classIdx), classAcc);
end

%% Confusion Matrix
figure('Name', 'Confusion Matrix', 'NumberTitle', 'off', 'Position', [100 100 700 600]);
cm = confusionchart(YTrue, YPred, ...
    'Title', sprintf('Confusion Matrix (Accuracy: %.2f%%)', accuracy), ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
cm.FontSize = 11;

%% Sample predictions visualization
figure('Name', 'Sample Predictions', 'NumberTitle', 'off', 'Position', [100 100 900 500]);
numSamples = min(20, numel(imdsTest.Files));
idx = randperm(numel(imdsTest.Files), numSamples);

for k = 1:numSamples
    img = readimage(imdsTest, idx(k));
    subplot(4, 5, k);
    imshow(img);

    isCorrect = YPred(idx(k)) == YTrue(idx(k));
    color = 'g';
    if ~isCorrect; color = 'r'; end

    title(sprintf('T:%s P:%s', char(YTrue(idx(k))), char(YPred(idx(k)))), ...
        'Color', color, 'FontSize', 8);
end
sgtitle('Sample Predictions (Green=Correct, Red=Wrong)', 'FontSize', 12);

fprintf('\nEvaluation complete.\n');

end


%% Fallback: evaluate on 4D array
function evaluate_model_array(net, XTest, YTest)

YPred = classify(net, XTest);
YTrue = categorical(YTest);
accuracy = sum(YPred == YTrue) / numel(YTrue) * 100;
fprintf('Test Accuracy: %.2f%%\n', accuracy);

figure;
confusionchart(YTrue, YPred, 'Title', sprintf('Confusion Matrix (Acc: %.2f%%)', accuracy));

end
