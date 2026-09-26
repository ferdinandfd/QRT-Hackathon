import os, sys
import numpy as np
import pandas as pd

RET_PREFIX = "RET_"
WINDOW = 20
AUTOCORR_LAGS = (1, 2, 3, 4, 5)
EWMA_HALFLIVES = (2, 5, 10)
ADD_EW_VOL = True
EW_VOL_HALFLIFE = 5


def get_ret_cols(df, k=WINDOW):
    cols = [f"{RET_PREFIX}{i}" for i in range(1, k + 1)]
    present = [c for c in cols if c in df.columns]
    if not present:
        raise KeyError(f"No return columns {RET_PREFIX}1..{k} found.")
    if len(present) < k:
        missing = [c for c in cols if c not in present]
        print(f"[Warn] missing columns ignored: {missing}")
    return present


def row_mean_range(row, start, end):
    vals = [row.get(f"{RET_PREFIX}{i}") for i in range(start, end + 1)]
    vals = [v for v in vals if pd.notna(v)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def row_up_streak(row, k=WINDOW):
    streak = 0
    for i in range(1, k + 1):
        v = row.get(f"{RET_PREFIX}{i}")
        if pd.isna(v) or v <= 0:
            break
        streak += 1
    return streak


def row_arg_pos(row, func=np.argmax, k=WINDOW):
    vals = [row.get(f"{RET_PREFIX}{i}") for i in range(1, k + 1)]
    vals = np.array([np.nan if v is None else v for v in vals], dtype=float)
    if np.all(np.isnan(vals)):
        return np.nan
    idx = func(vals)  # 0-based
    return idx + 1  # 1..k (1 = most recent)


def row_slope_trend(row, k=WINDOW):

    ys = []
    xs = []
    for i in range(1, k + 1):
        v = row.get(f"{RET_PREFIX}{i}")
        if pd.notna(v):
            xs.append(i)
            ys.append(v)
    if len(ys) < 2:
        return np.nan
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    x_mean = xs.mean()
    y_mean = ys.mean()
    denom = np.sum((xs - x_mean) ** 2)
    if denom == 0:
        return np.nan
    slope = np.sum((xs - x_mean) * (ys - y_mean)) / denom
    return float(slope)


def row_autocorr(row, lag, k=WINDOW):
    xs = []
    ys = []
    for i in range(1, k - lag + 1):
        a = row.get(f"{RET_PREFIX}{i}")
        b = row.get(f"{RET_PREFIX}{i+lag}")
        if pd.notna(a) and pd.notna(b):
            xs.append(a)
            ys.append(b)
    if len(xs) < 2:
        return np.nan
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    x -= x.mean()
    y -= y.mean()
    denom = np.sqrt((x**2).sum() * (y**2).sum())
    if denom == 0:
        return np.nan
    return float((x * y).sum() / denom)


def add_time_features(df, name):
    ret_cols = get_ret_cols(df, WINDOW)

    for hl in EWMA_HALFLIVES:

        w = np.array([0.5 ** ((i - 1) / hl) for i in range(1, WINDOW + 1)], dtype=float)
        w = w / w.sum()
        vals = df[ret_cols].to_numpy(float)

        isnan = np.isnan(vals)
        eff_w_sum = (~isnan) * w
        eff_w_sum = eff_w_sum.sum(axis=1)
        filled = np.where(isnan, 0.0, vals)
        num = (filled * w).sum(axis=1)
        ewma = np.divide(
            num, eff_w_sum, out=np.full_like(num, np.nan), where=eff_w_sum > 0
        )
        df[f"RET_EWMA_HL{hl}"] = ewma

    df["RET_RECENT_MEAN_3"] = df.apply(
        lambda r: row_mean_range(r, 1, min(3, WINDOW)), axis=1
    )
    tail_start = max(10, (WINDOW // 2) + 1)
    df["RET_OLD_MEAN"] = df.apply(
        lambda r: row_mean_range(r, tail_start, WINDOW), axis=1
    )
    df["RET_RECENT_MINUS_OLD"] = df["RET_RECENT_MEAN_3"] - df["RET_OLD_MEAN"]

    df["RET_TREND_SLOPE"] = df.apply(row_slope_trend, axis=1)

    for L in AUTOCORR_LAGS:
        df[f"RET_ACF_L{L}"] = df.apply(lambda r, lag=L: row_autocorr(r, lag), axis=1)

    df["RET_UP_STREAK"] = df.apply(row_up_streak, axis=1)

    df["RET_ARGMAX_POS"] = df.apply(lambda r: row_arg_pos(r, np.argmax), axis=1)
    df["RET_ARGMIN_POS"] = df.apply(lambda r: row_arg_pos(r, np.argmin), axis=1)

    if ADD_EW_VOL:
        w = np.array(
            [0.5 ** ((i - 1) / EW_VOL_HALFLIFE) for i in range(1, WINDOW + 1)],
            dtype=float,
        )
        w = w / w.sum()
        vals = df[ret_cols].to_numpy(float)
        isnan = np.isnan(vals)
        filled = np.where(isnan, np.nan, vals)

        num = np.nansum(filled * w, axis=1)
        w_eff = np.sum((~isnan) * w, axis=1)
        mu = np.divide(num, w_eff, out=np.full_like(num, np.nan), where=w_eff > 0)

        diff2 = (filled - mu[:, None]) ** 2
        var = np.nansum(diff2 * w, axis=1) / w_eff
        df["RET_EW_VOL_HL5"] = np.sqrt(var)

    return df


def main():
    in_train, in_test = "X_train_01.csv", "X_test_01.csv"
    out_train, out_test = "X_train_time.csv", "X_test_time.csv"
    if not (os.path.exists(in_train) and os.path.exists(in_test)):
        raise FileNotFoundError(
            "Place X_train_01.csv and X_test_01.csv in the current directory."
        )

    print("[Info] Loading ...")
    Xtr = pd.read_csv(in_train)
    Xte = pd.read_csv(in_test)

    print("[Info] Building temporal features for train ...")
    Xtr = add_time_features(Xtr, "train")
    print("[Info] Building temporal features for test ...")
    Xte = add_time_features(Xte, "test")

    print(f"[Info] Saving -> {out_train}, {out_test}")
    Xtr.to_csv(out_train, index=False)
    Xte.to_csv(out_test, index=False)
    print(f"[Done] {out_train}: {Xtr.shape}")
    print(f"[Done] {out_test}:  {Xte.shape}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)
