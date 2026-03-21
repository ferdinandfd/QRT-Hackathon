import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

RANDOM_STATE = 42
N_SPLITS = 5

def safe_read_csv(path, required=True):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        else:
            print(f"[Warn] File not found (skipped): {path}")
            return None
    print(f"[Info] Loading {path} ...")
    return pd.read_csv(path)

def one_hot_encode_allocation(X_train, X_test):
    """
    One-hot sur ALLOCATION en combinant train+test pour aligner les colonnes.
    Retourne (X_train_ohe, X_test_ohe)
    """
    if "ALLOCATION" not in X_train.columns or "ALLOCATION" not in X_test.columns:
        raise KeyError("Column 'ALLOCATION' must be present in both X_train and X_test.")

    X_train_c = X_train.copy()
    X_test_c  = X_test.copy()

    
    combined = pd.concat([X_train_c, X_test_c], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=["ALLOCATION"], prefix="ALLOC", dtype=np.uint8)

    
    X_train_ohe = combined.iloc[:len(X_train_c)].copy()
    X_test_ohe  = combined.iloc[len(X_train_c):].copy()

    n_alloc_cols = sum(col.startswith('ALLOC_') for col in X_train_ohe.columns)
    print(f"[Info] One-hot: créé {n_alloc_cols} colonnes ALLOC_*.")    
    return X_train_ohe, X_test_ohe

def target_encode_ts_oof(X_train, y):
    """
    Target encoding OOF pour 'TS' sur le train.
    Retourne une Series alignée avec X_train.index contenant TS_ENC.
    """
    if "TS" not in X_train.columns:
        raise KeyError("Column 'TS' must be present in X_train for target encoding.")

    ts_enc = pd.Series(index=X_train.index, dtype=float)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    global_mean = y.mean()

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_train), start=1):
        ts_mean = (
            X_train.loc[tr_idx, ["TS"]]
            .assign(target=y.iloc[tr_idx].values)
            .groupby("TS")["target"]
            .mean()
        )
        enc_val = X_train.loc[va_idx, "TS"].map(ts_mean).fillna(global_mean)
        ts_enc.loc[va_idx] = enc_val.values
        print(f"[Info] OOF target encoding: fold {fold_id}/{N_SPLITS} ok.")

    return ts_enc

def target_encode_ts_for_test(X_train, y, X_test):
    """
    Mapping pour le test: moyennes 'TS' calculées sur tout le train.
    Fallback: moyenne globale si 'TS' inconnue.
    """
    global_mean = y.mean()
    ts_mean_full = (
        X_train[["TS"]].assign(target=y.values)
        .groupby("TS")["target"]
        .mean()
    )
    return X_test["TS"].map(ts_mean_full).fillna(global_mean)

def main():
    X_train = safe_read_csv("X_train.csv", required=True)
    y_df    = safe_read_csv("y_train.csv", required=True)
    X_test  = safe_read_csv("X_test.csv", required=True)

    
    if "ROW_ID" not in X_train.columns or "ROW_ID" not in X_test.columns:
        raise KeyError("Both X_train.csv and X_test.csv must contain 'ROW_ID'.")
    if not {"ROW_ID", "target"}.issubset(y_df.columns):
        raise KeyError("y_train.csv must contain columns: 'ROW_ID' and 'target'.")

   
    Xy = X_train.merge(y_df[["ROW_ID", "target"]], on="ROW_ID", how="inner")
    if len(Xy) == 0:
        raise ValueError("Merge on ROW_ID produced 0 rows. Check ROW_ID overlap between X_train and y_train.")

    
    y = Xy["target"].reset_index(drop=True)
    X_train_aligned = Xy.drop(columns=["target"]).reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

   
    X_train_ohe, X_test_ohe = one_hot_encode_allocation(X_train_aligned, X_test)

   
    ts_enc_train = target_encode_ts_oof(X_train_ohe, y)
    X_train_ohe["TS_ENC"] = ts_enc_train

  
    ts_enc_test = target_encode_ts_for_test(X_train_ohe, y, X_test_ohe)
    X_test_ohe["TS_ENC"] = ts_enc_test

    
    if "TS" in X_train_ohe.columns:
        X_train_ohe = X_train_ohe.drop(columns=["TS"])
    if "TS" in X_test_ohe.columns:
        X_test_ohe = X_test_ohe.drop(columns=["TS"])

   
    non_numeric_train = X_train_ohe.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric_test  = X_test_ohe.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_train:
        print(f"[Warn] Non-numeric columns in train: {non_numeric_train}")
    if non_numeric_test:
        print(f"[Warn] Non-numeric columns in test: {non_numeric_test}")

   
    out_train = "X_train_01.csv"
    out_test  = "X_test_01.csv"
    X_train_ohe.to_csv(out_train, index=False)
    X_test_ohe.to_csv(out_test, index=False)

    print(f"[Done] Saved: {out_train} ({X_train_ohe.shape[0]} rows, {X_train_ohe.shape[1]} cols)")
    print(f"[Done] Saved: {out_test}  ({X_test_ohe.shape[0]} rows,  {X_test_ohe.shape[1]} cols)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)
