"""
Script to compare different FICO score bucketing (quantization) methods
by maximizing the total log-likelihood of the buckets.

This script is a conversion of the task_4_FICO_score_bucketing.ipynb notebook,
with all visualization code removed and a final comparison added.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Add the source directory to the Python path to import load_loan_data
try:
    sys.path.append('../src')
    from task_3_credit_risk_analysis import load_loan_data
except ImportError:
    print("Error: Could not import 'load_loan_data' from '../src'.")
    print("Please ensure 'task_3_credit_risk_analysis.py' is in the 'src' directory.")
    sys.exit(1)

# --- Core Function Definitions ---

def log_likelihood(k, n, eps=1e-9):
    """Compute the log-likelihood of defaults given k defaults out of n."""
    # Handle edge case where a bucket might have n=0
    if n == 0:
        return 0.0
    
    p = np.clip(k / n, eps, 1 - eps)
    return k * np.log(p) + (n - k) * np.log(1 - p)

def supervised_bucketing(fico_scores, defaults, n_buckets):
    """
    Finds boundaries using a Decision Tree (a 'supervised' method).
    """
    model = DecisionTreeClassifier(max_leaf_nodes=n_buckets)
    model.fit(fico_scores.reshape(-1, 1), defaults)
    boundaries = np.sort(model.tree_.threshold[model.tree_.threshold > 0])
    return boundaries

def optimal_binning_loglik(fico_score, default, n_bins):
    """
    Simple dynamic programming version (O(n_bins * U^2)) to find fico_score bucket boundaries
    maximizing total log-likelihood.
    """
    # Sort and group by unique scores
    df_grouped = pd.DataFrame({'fico_score': fico_score, 'default': default}).sort_values('fico_score')
    grouped = df_grouped.groupby('fico_score')['default'].agg(['count', 'sum']).reset_index()
    scores = grouped['fico_score'].values
    n = grouped['count'].values
    k = grouped['sum'].values
    U = len(scores)

    # cumulative sums
    cum_n = np.concatenate([[0], np.cumsum(n)])
    cum_k = np.concatenate([[0], np.cumsum(k)])

    # helper: log-likelihood of interval (i, j]
    def interval_ll(i, j):
        n_ij = cum_n[j] - cum_n[i]
        k_ij = cum_k[j] - cum_k[i]
        return log_likelihood(k_ij, n_ij)

    # DP arrays
    dp = np.full((n_bins + 1, U + 1), -np.inf)
    split = np.full((n_bins + 1, U + 1), -1, dtype=int)

    # base case: 1 bin
    for j in range(1, U + 1):
        dp[1, j] = interval_ll(0, j)

    # fill table
    for b in range(2, n_bins + 1):
        for j in range(b, U + 1):
            best_val, best_i = -np.inf, -1
            for i in range(b - 1, j):
                val = dp[b - 1, i] + interval_ll(i, j)
                if val > best_val:
                    best_val, best_i = val, i
            dp[b, j] = best_val
            split[b, j] = best_i

    # backtrack optimal boundaries
    boundaries_idx = []
    j, b = U, n_bins
    while b > 1:
        i = split[b, j]
        boundaries_idx.append(i)
        j, b = i, b - 1
    boundaries_idx = boundaries_idx[::-1]

    # convert to fico_score thresholds
    # Handle edge case where boundary index might be 0
    thresholds = []
    for i in boundaries_idx:
        if i == 0:
            # Not enough data to split, or an issue, use the first score
            thresholds.append(scores[0])
        else:
            thresholds.append((scores[i-1] + scores[i]) / 2)
            
    return thresholds

def calculate_total_loglik(df, bucket_col):
    """
    Calculates the total log-likelihood for a given set of buckets.
    """
    # Group by the new buckets to get n (count) and k (sum of defaults)
    bucket_stats = df.groupby(bucket_col, observed=True)['default'].agg(
        n='count',
        k='sum'
    )
    
    # Calculate log-likelihood for each bucket and sum them up
    total_ll = bucket_stats.apply(
        lambda row: log_likelihood(row['k'], row['n']),
        axis=1
    ).sum()
    
    return total_ll

# --- Main Execution ---

def main():
    """
    Main function to load data and compare bucketing methods.
    """
    print("Loading loan data...")
    df = load_loan_data('data/Task 3 and 4_Loan_Data.csv')
    
    n_buckets = 8
    print(f"Comparing 4 bucketing methods for {n_buckets} buckets...\n")
    
    # --- 1. Equal-Width Bucketing ---
    df['bucket_eq_width'] = pd.cut(df['fico_score'], bins=n_buckets, labels=False)
    ll_eq_width = calculate_total_loglik(df, 'bucket_eq_width')
    
    # --- 2. Equal-Frequency Bucketing ---
    # duplicates='drop' is necessary as FICO scores are discrete
    df['bucket_eq_freq'] = pd.qcut(df['fico_score'], q=n_buckets, labels=False, duplicates='drop')
    ll_eq_freq = calculate_total_loglik(df, 'bucket_eq_freq')

    # --- 3. Supervised Bucketing (Decision Tree) ---
    boundaries_sup = supervised_bucketing(df['fico_score'].values, df['default'].values, n_buckets)
    bins_sup = [-np.inf] + list(boundaries_sup) + [np.inf]
    df['bucket_sup'] = pd.cut(df['fico_score'], bins=bins_sup, labels=False, duplicates='drop')
    ll_sup = calculate_total_loglik(df, 'bucket_sup')
    
    # --- 4. Optimal Bucketing (Dynamic Programming) ---
    print("Running Dynamic Programming... (this may take a moment)")
    thresholds_dp = optimal_binning_loglik(df['fico_score'].values, df['default'].values, n_bins=n_buckets)
    bins_dp = [-np.inf] + thresholds_dp + [np.inf]
    df['bucket_dp'] = pd.cut(df['fico_score'], bins=bins_dp, labels=False, duplicates='drop')
    ll_dp = calculate_total_loglik(df, 'bucket_dp')
    
    print("Optimal (DP) thresholds found:", thresholds_dp)
    print("\n--- Comparison of Total Log-Likelihood (Higher is Better) ---")
    
    results = {
        "Equal-Width": ll_eq_width,
        "Equal-Frequency": ll_eq_freq,
        "Supervised (Tree)": ll_sup,
        "Optimal (DP)": ll_dp,
    }
    
    # Sort results for clear comparison
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    
    for method, score in sorted_results:
        print(f"{method:>20}: {score:.4f}")
        
    print(f"\n🏆 Best Method: {sorted_results[0][0]}")

if __name__ == "__main__":
    main()