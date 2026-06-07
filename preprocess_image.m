function imgOut = preprocess_image(imgIn)
%PREPROCESS_IMAGE Preprocess an input image for digit classification
%
%   imgOut = preprocess_image(imgIn) takes a raw image (grayscale or RGB,
%   any size) and returns a normalized 28x28 grayscale image ready for
%   classification.
%
%   Steps:
%       1. Convert to grayscale (if RGB)
%       2. Resize to 28x28
%       3. Invert if background is white (MNIST uses black background)
%       4. Normalize pixel values to [0, 1]
%
%   Input:
%       imgIn  - Raw image (uint8 or double, grayscale or RGB)
%   Output:
%       imgOut - Preprocessed 28x28 single-channel image (double)

%% 1. Convert to grayscale
if size(imgIn, 3) == 3
    imgGray = rgb2gray(imgIn);
else
    imgGray = imgIn;
end

%% 2. Convert to double
imgGray = double(imgGray);

%% 3. Resize to 28x28
imgResized = imresize(imgGray, [28, 28]);

%% 4. Normalize to [0, 1]
imgNorm = imgResized / 255.0;

%% 5. Invert if background is light (white paper = bright background)
%    MNIST digits have WHITE digit on BLACK background
meanVal = mean(imgNorm(:));
if meanVal > 0.5
    imgNorm = 1 - imgNorm;  % Invert
end

%% 6. Threshold (optional binarization for cleaner input)
% imgNorm = imbinarize(imgNorm);  % Uncomment if needed

imgOut = imgNorm;

end
