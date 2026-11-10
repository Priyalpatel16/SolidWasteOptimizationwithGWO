# Optimizing Solid Waste Classification Using Deep Learning and Grey Wolf Optimizer (GWO)s

## Overview

This project explores image-based waste classification using **EfficientNet** and the **Grey Wolf Optimizer (GWO)**. The objective is to classify waste images into six categories and study how GWO-based optimization and different class-weighting strategies affect classification performance.

The project contains a controlled comparison of six model configurations. All six models were trained using the **same training dataset** and evaluated on the **same balanced test dataset**, with an equal number of images from each class.

## Dataset

The project uses the [TrashNet dataset](https://github.com/garythung/trashnet), which contains images belonging to six waste categories:

1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic
6. Trash

### Evaluation protocol

- The same training data was used for all six model configurations.
- All six models were evaluated on the same test dataset.
- The test dataset was balanced: it contained an equal number of images from each of the six classes.
- Accuracy, Macro-F1 and Weighted-F1 were used for comparison.

Using a balanced test set makes the evaluation less dominated by any one class. Because every class has equal support in this test set, **Macro-F1 and Weighted-F1 are expected to be equal (apart from possible rounding)**. They are both reported for completeness, but they should not be interpreted as two independent improvements.

## Methodology

The experiments combine two choices:

- **Model optimization:** EfficientNet trained in its standard configuration or with hyperparameter optimization using the Grey Wolf Optimizer.
- **Class weighting:** no class weights, inverse-frequency weights, or direct-frequency weights.

The class-weighting strategies are:

- **No class weights:** the standard classification loss is used without class-specific weighting.
- **Inverse-frequency weights:** classes with fewer training examples receive larger weights, giving minority classes more influence on the loss.
- **Direct-frequency weights:** class weights are proportional to their frequency, giving more influence to classes with more training examples.

Combining the two model-optimization choices with the three weighting strategies gives the six configurations below.

## Model Comparison

All values below are percentages and are from the same balanced test set.

| Model | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) |
|---|---:|---:|---:|
| **GWO-EfficientNet – Inverse-frequency weights** | **88.27** | **88.25** | **88.25** |
| GWO-EfficientNet – No class weights | 85.19 | 84.92 | 84.92 |
| EfficientNet – Inverse-frequency weights | 80.25 | 80.35 | 80.35 |
| GWO-EfficientNet – Direct-frequency weights | 79.63 | 78.60 | 78.60 |
| EfficientNet – No class weights | 77.78 | 77.24 | 77.24 |
| EfficientNet – Direct-frequency weights | 74.69 | 72.52 | 72.52 |

### Key observations

- **Best-performing configuration:** GWO-EfficientNet with inverse-frequency class weights achieved **88.27% accuracy** and **88.25% Macro-F1**.
- **Effect of GWO with inverse-frequency weights:** accuracy increased from 80.25% to 88.27%, a gain of **8.02 percentage points**.
- **Effect of GWO without class weights:** accuracy increased from 77.78% to 85.19%, a gain of **7.41 percentage points**.
- **Effect of GWO with direct-frequency weights:** accuracy increased from 74.69% to 79.63%, a gain of **4.94 percentage points**.
- In this comparison, inverse-frequency weighting performed best, while direct-frequency weighting produced the lowest scores.

These findings apply to the training setup and balanced test set used in this project. They should not be treated as a direct reproduction or head-to-head benchmark of the accuracy reported in the reference paper, because the experimental protocols and evaluation data may differ.

## Additional Experiments and Analysis

Beyond implementing the central EfficientNet-and-GWO approach, this project includes the following project-specific comparisons and evaluation steps:

1. **Three-way class-weighting study:** inverse-frequency weighting, direct-frequency weighting and no class weights were compared.
2. **GWO versus non-GWO comparison:** each class-weighting strategy was evaluated both with and without GWO, allowing the effect of optimization to be examined under matched weighting conditions.
3. **Balanced test-set evaluation:** every class was represented by the same number of test images, supporting a more even class-wise evaluation.
4. **Multiple evaluation metrics:** accuracy, Macro-F1 and Weighted-F1 were reported rather than relying on accuracy alone.
5. **Class-level error analysis:** classification reports and confusion matrices were also generated during model diagnostics to inspect class-wise precision, recall, F1-score and misclassifications.

This README describes the six-model comparison above. It intentionally does **not** include the later experimental fine-tuning work or the subsequent inverse-weight fine-tuning variants.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Reference Research Paper

Eryeşil, Y., Kahramanli Örnek, H., & Taşdemir, Ş. *Optimizing solid waste classification using deep learning and grey wolf optimizer for recycling efficiency*. **International Journal of Environmental Science and Technology**, 23, 44 (2026).

[https://doi.org/10.1007/s13762-025-06834-2](https://doi.org/10.1007/s13762-025-06834-2)

## Future Scope

- Evaluate the classifier on additional waste datasets and images captured in real-world environments.
- Investigate lightweight models for faster inference on edge devices.
- Explore deployment in an automated waste-segregation pipeline.
 
