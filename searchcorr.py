import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("X_train_ready.csv")
if "target" not in df.columns:
    raise ValueError("X_train_ready.csv must contain 'target'")

df = df.apply(pd.to_numeric, errors="coerce")
features = df.select_dtypes(include=[np.number]).drop(columns=["target"])

correlations = features.corrwith(df["target"]).dropna()
correlations_sorted = correlations.reindex(
    correlations.abs().sort_values(ascending=False).index
)

correlations_df = correlations_sorted.reset_index()
correlations_df.columns = ["feature", "correlation"]
correlations_df.to_csv("feature_target_correlations.csv", index=False)
print("[OK] CSV file saved → feature_target_correlations.csv")

print("\nTop 30 features most correlated with the target:\n")
print(correlations_sorted.head(30).round(6))

top_feats = correlations_sorted.head(20).index.tolist()
corr_matrix = df[top_feats + ["target"]].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlations among the top 20 features and the target")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
print("[OK] Heatmap saved → correlation_heatmap.png")
