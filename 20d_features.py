import pandas as pd
import numpy as np
from pathlib import Path

TRAIN_PATH = "X_train_time.csv"
TEST_PATH  = "X_test_time.csv"
Y_PATH     = "y_train.csv"

OUT_TRAIN  = "X_train_ready.csv"
OUT_TEST   = "X_test_ready.csv"
OUT_FEATS  = "features.txt"   # ordre de colonnes utilisé par le script d'entrainement


Xtr = pd.read_csv(TRAIN_PATH)
Xte = pd.read_csv(TEST_PATH)
ydf = pd.read_csv(Y_PATH)


for df, name in [(Xtr, "X_train_time.csv"), (Xte, "X_test_time.csv")]:
    if "ROW_ID" not in df.columns:
        raise ValueError(f"{name} doit contenir 'ROW_ID'.")
if "ROW_ID" not in ydf.columns or "target" not in ydf.columns:
    raise ValueError("y_train.csv doit contenir 'ROW_ID' et 'target'.")


data = Xtr.merge(ydf[["ROW_ID", "target"]], on="ROW_ID", how="inner")
if len(data) == 0:
    raise ValueError("Merge sur ROW_ID → 0 ligne. Vérifie la correspondance des ROW_ID.")



NON_FEATS = {"ROW_ID", "target", "TS", "DATE", "date", "timestamp", "ALLOCATION"}
feat_cols_train = [c for c in data.columns if c not in NON_FEATS]


for c in feat_cols_train:
    if c not in Xte.columns:
        Xte[c] = np.nan
Xte_aligned = Xte[["ROW_ID"] + feat_cols_train].copy()


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.columns:
        if c != "ROW_ID":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

data_clean = data[["ROW_ID", "target"] + feat_cols_train].copy()
data_clean = clean_numeric(data_clean)
Xte_clean  = clean_numeric(Xte_aligned)


data_clean.to_csv(OUT_TRAIN, index=False)
Xte_clean.to_csv(OUT_TEST, index=False)


Path(OUT_FEATS).write_text("\n".join(feat_cols_train), encoding="utf-8")

print(f"[OK] {OUT_TRAIN} et {OUT_TEST} écrits.")
print(f"[OK] {OUT_FEATS} (features: {len(feat_cols_train)})")
