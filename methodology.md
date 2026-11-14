# Methodology

AURELIS is an educational binary image-classification experiment using transfer learning with ResNet-18. Images are resized to 224×224 and normalized using ImageNet channel statistics. Random horizontal flipping is applied only to the training subset. A stratified validation subset is drawn from the training directory with a fixed random seed. The separately supplied test directory is reserved for final evaluation.

The evaluation script reports accuracy, precision, recall/sensitivity, F1, specificity, and confusion-matrix counts (TN, FP, FN, TP). It also reports an always-predict-PNEUMONIA baseline so model performance can be compared with a simple reference. Do not use the test split for tuning.

This benchmark workflow does not establish clinical safety, fairness, calibration, or utility. Dataset provenance, patient-level separation, duplicate images, and label quality require independent review.
