# Handwritten Digit Classification in MATLAB

A MATLAB-based handwritten digit classifier using a Convolutional Neural Network (CNN), trained on the MNIST-style digit dataset.

## Features
- CNN architecture with 3 convolutional blocks + fully connected layers
- Batch normalization and dropout for regularization
- Training progress visualization
- Confusion matrix and per-class accuracy reporting
- Single image prediction with confidence score

## Requirements
- MATLAB R2019b or later
- Deep Learning Toolbox

## Project Structure
```
├── main.m                  → Run this to train and evaluate
├── train_model.m           → CNN architecture and training
├── preprocess_image.m      → Image preprocessing pipeline
├── predict_digit.m         → Predict a single digit image
├── evaluate_model.m        → Accuracy + confusion matrix
├── load_builtin_dataset.m  → Load MATLAB's built-in dataset
└── sample_images/          → Place test images here
```

## How to Run

1. Open MATLAB and set the project folder as your working directory
2. Run the main script:
```matlab
main
```
3. To predict a single image after training:
```matlab
load('trained_digit_classifier.mat');
[label, conf] = predict_digit(net, 'sample_images/my_digit.png');
```

## Model Architecture
| Layer | Type | Details |
|-------|------|---------|
| 1 | Input | 28×28×1, zero-center normalization |
| 2–4 | Conv Block 1 | 8 filters, 3×3, BN, ReLU, MaxPool |
| 5–7 | Conv Block 2 | 16 filters, 3×3, BN, ReLU, MaxPool |
| 8–10 | Conv Block 3 | 32 filters, 3×3, BN, ReLU |
| 11 | FC | 128 units, ReLU |
| 12 | Dropout | p = 0.5 |
| 13 | FC | 10 units (classes 0–9) |
| 14 | Output | Softmax + Classification |

## Results
- Expected test accuracy: **~98–99%** on MATLAB's built-in digit dataset
- Training time: ~2–5 minutes on CPU

## Author
[Your Name] — [Your Institution]
