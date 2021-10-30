#!/bin/bash

# create directory if not yet existing
mkdir -p data/classification/

# run feature extraction on training set (may need to fit extractors)
echo "  training set"
python -m code.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle --random_forest -rf_n_estimators 50 -rf_criterion "entropy" -rf_depth 50 -rf_bootstrap "True" -rf_class_weight "balanced" -s 42 --accuracy --kappa --average_precision --f1_score

# run feature extraction on validation and test set (with pre-fit extractors)
echo "  validation set"
python -m code.classification.run_classifier data/dimensionality_reduction/validation.pickle -i data/classification/classifier.pickle --accuracy --kappa --average_precision --f1_score
echo "  test set"
python -m code.classification.run_classifier data/dimensionality_reduction/test.pickle -i data/classification/classifier.pickle --accuracy --kappa --average_precision --f1_score
