import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

TRAIN_READY = "X_train_ready.csv"
TEST_READY = "X_test_ready.csv"
FEATS_FILE = "features.txt"

N_SPLITS = 10
SEED = 42


train = pd.read_csv(TRAIN_READY)
test = pd.read_csv(TEST_READY)
feat_cols = Path(FEATS_FILE).read_text(encoding="utf-8").splitlines()


for col in ["ROW_ID", "target"]:
    if col not in train.columns:
        raise ValueError(f"{TRAIN_READY} must contain '{col}'.")
if "ROW_ID" not in test.columns:
    raise ValueError(f"{TEST_READY} must contain 'ROW_ID'.")


X = train[feat_cols].astype(np.float32).values
y = (train["target"].astype(float) > 0).astype(int).values
X_te = test[feat_cols].astype(np.float32).values


model = ExtraTreesClassifier(
    n_estimators=600,
    max_depth=15,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=SEED,
    n_jobs=-1,
)

pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", model)])


kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
accs = []

for i, (tr, va) in enumerate(kf.split(X, y), 1):
    pipe.fit(X[tr], y[tr])
    p = pipe.predict_proba(X[va])[:, 1]
    pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y[va], pred)
    accs.append(acc)
    print(f"Fold {i:02d} | Acc@0.5 = {acc:.4f}")

print(f"\nMean Acc (StratifiedKFold {N_SPLITS}): {np.mean(accs):.4f}")


pipe.fit(X, y)


proba = pipe.predict_proba(X_te)[:, 1]
yhat = (proba >= 0.5).astype(int)

submission = pd.DataFrame({"ROW_ID": test["ROW_ID"].values, "target": yhat})

if len(submission) != 7735:
    print(f"[Warning] submission has {len(submission)} rows; expected 7735.")

submission.to_csv("submission.csv", index=False)
print(
    f"[OK] submission.csv saved — shape={submission.shape} | "
    f"positive_predictions={submission['target'].sum()} ({submission['target'].mean():.2%})"
)
