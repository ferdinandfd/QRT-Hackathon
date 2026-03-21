
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


X_train = pd.read_csv("X_train_ready.csv")
X_test  = pd.read_csv("X_test_ready.csv")


non_feats = {"ROW_ID", "target"}
features = [col for col in X_train.columns if col not in non_feats]


for feat in features:
    if feat not in X_test.columns:
        continue

    plt.figure(figsize=(8, 4))
    sns.kdeplot(X_train[feat], label="Train", fill=True)
    sns.kdeplot(X_test[feat], label="Test", fill=True)
    plt.title(f"Distribution: {feat}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"distrib_{feat}.png")
    plt.close()

print("[OK] Distributions sauvegardées (format: distrib_<feature>.png)")
