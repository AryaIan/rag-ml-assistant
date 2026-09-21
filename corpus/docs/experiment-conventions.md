# Experiment conventions

## Evaluation protocol
Every experiment uses stratified 5-fold cross-validation with shuffling and seed 1234.
Reported scores are the mean and standard deviation across the 5 folds, in percentage points.
Runs without a fixed seed are not comparable.

## Metrics
Accuracy is the default metric. Macro F1 is added for multi-class model comparisons, because it
weights every class equally. Fit time is the mean seconds to train one fold. It depends on the
machine, so compare fit times only within a single report.

## Datasets
Two datasets bundled with scikit-learn are used, so no downloads are needed:
digits (1797 samples, 64 features, 10 classes) and breast cancer Wisconsin
(569 samples, 30 features, 2 classes).

## Interpreting differences
If the gap between the two best configurations is smaller than the fold-to-fold standard
deviation of the best one, the two are treated as not clearly different. Do not tune
hyperparameters on data you also report final results on.

## Reproducing
Run `python scripts/generate_experiments.py`. Accuracy is deterministic for a given seed and
scikit-learn version.
