import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


path_to_data = "./Data"
n_signal = 50000
n_background = 50000
sigma = 0.5
r_ring = 1.8

# Signal: 2D Gaussian
x_signal = np.random.normal(0, sigma, n_signal)
y_signal = np.random.normal(0, sigma, n_signal)

# Background: Ring (radius distributed as Gaussian around r_ring)
theta_bg = np.random.uniform(0, 2*np.pi, n_background)
radius_bg = np.random.normal(r_ring, sigma, n_background)
x_background = radius_bg * np.cos(theta_bg)
y_background = radius_bg * np.sin(theta_bg)

# Plot
plt.figure(figsize=(7,7))
plt.scatter(x_signal, y_signal, s=8, label='Signal', alpha=0.6)
plt.scatter(x_background, y_background, s=8, label='Background', alpha=0.6)
plt.gca().set_aspect('equal', 'box')
plt.legend()
plt.title('Signal (2D Gaussian) vs. Background (Ring)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


def print_feature_stats(X, label):
    print(f"\nFeature stats for {label}:")
    # If X is 1D, treat as a single feature
    if X.ndim == 1:
        min_val = np.min(X)
        max_val = np.max(X)
        hist, bin_edges = np.histogram(X, bins=10)
        print(f"Feature 0: min={min_val:.4f}, max={max_val:.4f}")
        print(f"  Histogram bins: {hist}")
        print(f"  Bin edges: {bin_edges}\n")
    else:
        for i in range(X.shape[1]):
            feature = X[:, i]
            min_val = np.min(feature)
            max_val = np.max(feature)
            hist, bin_edges = np.histogram(feature, bins=10)
            print(f"Feature {i}: min={min_val:.4f}, max={max_val:.4f}")
            print(f"  Histogram bins: {hist}")
            print(f"  Bin edges: {bin_edges}\n")

# For background, print stats for both x and y as features
bg_features = np.stack([x_background, y_background], axis=1)
print_feature_stats(bg_features, "Original BG")

# Print linear correlations
corr_signal = np.corrcoef(x_signal, y_signal)[0, 1]
corr_background = np.corrcoef(x_background, y_background)[0, 1]
print(f"Signal linear correlation (x, y): {corr_signal:.4f}")
print(f"Background linear correlation (x, y): {corr_background:.4f}")

# Print total correlation (mutual information approximation)
from sklearn.metrics import mutual_info_score

def total_correlation(x, y, bins=50):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    h_x = mutual_info_score(None, None, contingency=np.histogram(x, bins)[0].reshape(-1, 1))
    h_y = mutual_info_score(None, None, contingency=np.histogram(y, bins)[0].reshape(-1, 1))
    return h_x + h_y - mi

tc_signal = total_correlation(x_signal, y_signal)
tc_background = total_correlation(x_background, y_background)
print(f"Signal total correlation (approx): {tc_signal:.4f}")
print(f"Background total correlation (approx): {tc_background:.4f}")


signal_df = pd.DataFrame({'x1': x_signal, 'x2': y_signal, 'Label': 0})
background_df = pd.DataFrame({'x1': x_background, 'x2': y_background, 'Label': 1})
df = pd.concat([signal_df, background_df], ignore_index=True)

df.to_csv(f'{path_to_data}/donut_signal_background.csv', index=False)
print(f"Saved labeled events to {path_to_data}/donut_signal_background.csv")