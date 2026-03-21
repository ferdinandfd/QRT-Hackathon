import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns

# === Chargement des fichiers ===
X_train = pd.read_csv("X_train_ready.csv")
X_test = pd.read_csv("X_test_ready.csv")

# Ne garder que les features numériques communes
common_cols = X_train.columns.intersection(X_test.columns)
num_cols = X_train[common_cols].select_dtypes(include=np.number).columns

print(f"[Info] {len(num_cols)} colonnes numériques communes comparées.")

# === Calcul des distances de Wasserstein ===
shift_scores = []

for col in num_cols:
    train_vals = X_train[col].dropna()
    test_vals = X_test[col].dropna()
    dist = wasserstein_distance(train_vals, test_vals)
    shift_scores.append((col, dist))

# === Résultats dans un DataFrame ===
df_shift = pd.DataFrame(shift_scores, columns=["feature", "shift_score"])
df_shift = df_shift.sort_values(by="shift_score", ascending=False)

# === Sauvegarde CSV ===
df_shift.to_csv("feature_shift_scores.csv", index=False)
print(f"[OK] feature_shift_scores.csv écrit ({len(df_shift)} features).")

# === Heatmap des 20 plus instables (optionnel) ===
top_features = df_shift.head(20)["feature"].tolist()
combined = pd.concat([
    X_train[top_features].assign(dataset="train"),
    X_test[top_features].assign(dataset="test")
])

# Melt pour seaborn
df_melted = pd.melt(combined, id_vars="dataset", var_name="feature", value_name="value")

plt.figure(figsize=(12, 8))
sns.boxplot(data=df_melted, x="feature", y="value", hue="dataset", fliersize=1)
plt.xticks(rotation=90)
plt.title("Top 20 des features avec le plus grand shift de distribution")
plt.tight_layout()
plt.savefig("shifted_features_boxplot.png")
print("[OK] Heatmap enregistrée → shifted_features_boxplot.png")
