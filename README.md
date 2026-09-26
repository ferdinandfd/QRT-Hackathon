# QRT Hackathon — Tabular Classification Pipeline

A team project for a QRT hackathon. The repository contains the code used to clean the provided tabular data, derive features from recent returns, inspect train/test differences, and train an ExtraTrees classifier. The competition data are not included here.

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| Clean and align the supplied files | [`data_cleaning.py`](data_cleaning.py) | `X_train_01.csv`, `X_test_01.csv` |
| Build return-based temporal features | [`time_features.py`](time_features.py) | `X_train_time.csv`, `X_test_time.csv` |
| Select and align numeric features | [`20d_features.py`](20d_features.py) | `X_train_ready.csv`, `X_test_ready.csv`, `features.txt` |
| Train and produce predictions | [`test4.py`](test4.py) | 10-fold validation scores and `submission.csv` |

The cleaning step one-hot encodes `ALLOCATION` and creates an out-of-fold target encoding for `TS` on the training data. Temporal features use 20 recent return columns (`RET_1` to `RET_20`), including exponentially weighted means and volatility, a trend slope, autocorrelations, and short-term streaks. The final model is an ExtraTrees classifier with median imputation.

Three optional scripts examine the prepared data: [`searchcorr.py`](searchcorr.py) ranks feature/target correlations, [`distribution_analysis.py`](distribution_analysis.py) plots train/test feature distributions, and [`wasserstein_distance.py`](wasserstein_distance.py) ranks features by the Wasserstein distance between train and test.

## Run locally

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the supplied `X_train.csv`, `X_test.csv`, and `y_train.csv` files in the repository root. The scripts expect `ROW_ID` in all three files, `target` in `y_train.csv`, and `ALLOCATION`, `TS`, and `RET_1` through `RET_20` in the feature files.

```bash
python data_cleaning.py
python time_features.py
python 20d_features.py
python test4.py
```

The analysis scripts can be run after `20d_features.py`. The raw data, intermediate CSVs, plots, and submission file are excluded from Git by `.gitignore`.

## Evaluation note

`test4.py` prints accuracy at a 0.5 probability threshold for each of ten stratified folds, then refits on the full training set. This is an exploratory validation result: `TS` target encoding is generated before the model's cross-validation split, so its reported mean accuracy should not be treated as an unbiased held-out score. The repository contains no saved competition score or data with which to reproduce one.
