# QRT-Hackathon
In this Hackathon, we followed a structured pipeline: first cleaning and standardizing raw data to ensure consistency, including encoding allocations and dates to avoid introducing artificial ordering. We then engineered temporal features from the last 20 returns to capture trends, volatility, and internal correlations.

The enriched dataset was aligned and filtered for training. We analyzed feature correlations and computed Wasserstein distances between train and test sets to detect distribution shifts, though adjustments based on these findings did not improve accuracy.

Our final model was an ExtraTrees ensemble, chosen for robustness. In hindsight, linear models may have been more suitable given initial results, and hyperparameter optimization was challenging due to dataset size.

This project was part of QRT’s hackathon, offering valuable technical challenges and insights into their research culture.
