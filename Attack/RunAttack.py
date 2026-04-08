import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
from keras.layers import Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Helpers import generate_adversarial_samples

def main():
    path_to_data = './Data'
    results_path = './Results'
    model_path = './Models'

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_adversaries", type=int, default=5000)
    args = parser.parse_args()


    os.makedirs(results_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # --- Load the dummy data (generated in ./Data/MakeDonutDummyData.py)---
    df = pd.read_csv(os.path.join(path_to_data, 'donut_signal_background.csv'))
    X = df[['x1', 'x2']].to_numpy()
    y = df['Label'].to_numpy()  # 0: signal, 1: background

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


    # --- Train a simple classifier (only if not already trained) ---
    model_file = os.path.join(model_path, 'best_model.keras')
    if os.path.exists(model_file):
        model = keras.models.load_model(model_file)
        print(f"Loaded existing model from {model_file}")
    else:
        model = keras.Sequential([
            Dense(32, input_dim=2, activation='relu', kernel_regularizer=regularizers.L1(0.001)),
            BatchNormalization(),
            Dense(16, activation='relu', kernel_regularizer=regularizers.L1(0.001)),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        saveModel = ModelCheckpoint(model_file, save_best_only=True, monitor='val_loss', mode='min')
        model.fit(
            X_train, y_train,
            batch_size=256,
            epochs=30,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True), saveModel],
            verbose=2
        )
        print(f"Trained and saved model to {model_file}")

    # --- Evaluate on test set ---
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")


    # --- Visualize decision boundary with point clouds ---

    # Get signal and background for plotting
    signal_mask = (y == 0)
    background_mask = (y == 1)
    x_signal = X[signal_mask]
    x_background = X[background_mask]

    # Subsample to at most n_subsample points per class for plotting
    n_subsample = 1000
    def subsample(arr, max_samples=n_subsample):
        if arr.shape[0] > max_samples:
            idx = np.random.choice(arr.shape[0], max_samples, replace=False)
            return arr[idx]
        return arr

    x_signal_plot = subsample(x_signal, n_subsample)
    x_background_plot = subsample(x_background, n_subsample)

    # Create meshgrid
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid, verbose=0).reshape(xx.shape)

    plt.figure(figsize=(7,7))
    plt.scatter(x_signal_plot[:,0], x_signal_plot[:,1], s=8, alpha=0.5, label='Signal')
    plt.scatter(x_background_plot[:,0], x_background_plot[:,1], s=8, alpha=0.5, label='Background')
    contour = plt.contour(xx, yy, probs, levels=[0.5], colors='k', linewidths=2)
    from matplotlib.lines import Line2D
    proxy = [Line2D([0], [0], color='k', linewidth=2, label='Decision boundary')]
    plt.legend(handles=[
        plt.Line2D([], [], marker='o', color='w', markerfacecolor='C0', markersize=8, alpha=0.5, label='Signal'),
        plt.Line2D([], [], marker='o', color='w', markerfacecolor='C1', markersize=8, alpha=0.5, label='Background'),
        proxy[0]
    ])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'signal_background_decision_boundary.pdf'))
    plt.show()


    # --- Select background test samples that are correctly classified (these will be used for the attack) ---
    bg_mask = (y_test == 1)
    bg_X_test = X_test[bg_mask]
    bg_y_test = y_test[bg_mask]
    bg_pred = (model.predict(bg_X_test) > 0.5).astype(int).flatten()
    correct_bg_mask = (bg_pred == bg_y_test)
    bg_X_test_correct = bg_X_test[correct_bg_mask]
    bg_y_test_correct = bg_y_test[correct_bg_mask]
    print(f"Correctly classified background samples: {bg_X_test_correct.shape[0]}")

    # --- Optional: Limit number of test background samples for adversarial generation ---
    max_test = args.num_adversaries
    if bg_X_test_correct.shape[0] > max_test:
        idx = np.random.choice(bg_X_test_correct.shape[0], max_test, replace=False)
        bg_X_test_correct = bg_X_test_correct[idx]
        bg_y_test_correct = bg_y_test_correct[idx]
        print(f"Subsampled to {max_test} correctly classified background samples for adversarial generation.")

    def print_feature_stats(X, label):
        print(f"\nFeature stats for {label}:")
        for i in range(X.shape[1]):
            feature = X[:, i]
            min_val = np.min(feature)
            max_val = np.max(feature)
            hist, bin_edges = np.histogram(feature, bins=10)
            print(f"Feature {i}: min={min_val:.4f}, max={max_val:.4f}")
            print(f"  Histogram bins: {hist}")
            print(f"  Bin edges: {bin_edges}\n")

    print_feature_stats(bg_X_test_correct, "Original BG")

    model_path = f'{model_path}/best_model.keras'

    print(bg_X_test_correct)
    print("Shapes of correctly classified background samples:")
    print(bg_X_test_correct.shape, bg_y_test_correct.shape)


    # --- Generate adversarial samples for these background samples ---
    adversarial_path = os.path.join(results_path, f'adversarial_samples_bg.feather')
    if os.path.exists(adversarial_path):
        adv_df = pd.read_feather(adversarial_path)
        advs = adv_df[['x1', 'x2']].to_numpy()
    else:
        advs = generate_adversarial_samples(
            x_test_correct=bg_X_test_correct,
            y_test_correct=bg_y_test_correct,
            model_weights_path=model_path,
            min_change=0.001,
            step=0.001,
            n_iterations=10,
            mask=None,
            num_bins=70,
            n_gpus=1,
            verbose=True,
            alpha=6.0,
            beta=1.0,
            save_dir=results_path,
            use_no_change=True,
            max_jsd_single_change=0.005,
            max_frob_single_change=0.05,
            save_results=False, 
            randomize_step=True,
            random_step_frac=0.5,
            num_candidates=150
        )
        adv_df = pd.DataFrame(advs, columns=['x1', 'x2'])
        adv_df['Label'] = bg_y_test_correct
        adv_df.to_feather(adversarial_path)

    # --- Check classifier predictions on adversarial samples ---
    adv_pred = (model.predict(advs, batch_size=1024) > 0.5).astype(int).flatten()
    fooling_ratio = np.mean(adv_pred != bg_y_test_correct)
    print(f"Fooling ratio (background misclassified as signal): {fooling_ratio:.4f}")

    # --- Plot 2D distributions before and after attack ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(bg_X_test_correct[:, 0], bg_X_test_correct[:, 1], c='blue', alpha=0.4, label='Original Background')
    plt.title('Original Background (Correctly Classified)')
    plt.axis('equal')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(advs[:, 0], advs[:, 1], c='red', alpha=0.4, label='Adversarial Background')
    plt.title('Adversarial Background')
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'adversarial_vs_original_background_2d.pdf'))
    plt.show()

    # --- Plot 1D marginals and correlation matrices ---
    def plot_marginals_and_corr(X1, X2, label1, label2, fname_prefix):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs[0, 0].hist(X1[:, 0], bins=50, alpha=0.6, label=label1)
        axs[0, 0].hist(X2[:, 0], bins=50, alpha=0.6, label=label2)
        axs[0, 0].set_title('x1 marginal')
        axs[0, 0].legend()
        axs[0, 1].hist(X1[:, 1], bins=50, alpha=0.6, label=label1)
        axs[0, 1].hist(X2[:, 1], bins=50, alpha=0.6, label=label2)
        axs[0, 1].set_title('x2 marginal')
        axs[0, 1].legend()
        im1 = axs[1, 0].imshow(np.corrcoef(X1, rowvar=False), vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar(im1, ax=axs[1, 0])
        axs[1, 0].set_title(f'{label1} Corr')
        im2 = axs[1, 1].imshow(np.corrcoef(X2, rowvar=False), vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar(im2, ax=axs[1, 1])
        axs[1, 1].set_title(f'{label2} Corr')
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'{fname_prefix}_marginals_corr.pdf'))
        plt.show()

    plot_marginals_and_corr(bg_X_test_correct, advs, "Original BG", "Adversarial BG", "bg_adversarial_comparison")

    # --- Print linear and total correlation for clean and adversarial sets ---
    from sklearn.metrics import mutual_info_score

    def total_correlation(x, y, bins=50):
        c_xy = np.histogram2d(x, y, bins)[0]
        mi = mutual_info_score(None, None, contingency=c_xy)
        h_x = mutual_info_score(None, None, contingency=np.histogram(x, bins)[0].reshape(-1, 1))
        h_y = mutual_info_score(None, None, contingency=np.histogram(y, bins)[0].reshape(-1, 1))
        return h_x + h_y - mi

    # Clean (correctly classified background)
    corr_clean = np.corrcoef(bg_X_test_correct[:, 0], bg_X_test_correct[:, 1])[0, 1]
    tc_clean = total_correlation(bg_X_test_correct[:, 0], bg_X_test_correct[:, 1])
    print(f"Clean background linear correlation (x1, x2): {corr_clean:.4f}")
    print(f"Clean background total correlation (approx): {tc_clean:.4f}")

    # Adversarial
    corr_adv = np.corrcoef(advs[:, 0], advs[:, 1])[0, 1]
    tc_adv = total_correlation(advs[:, 0], advs[:, 1])
    print(f"Adversarial background linear correlation (x1, x2): {corr_adv:.4f}")
    print(f"Adversarial background total correlation (approx): {tc_adv:.4f}")

    # --- Point cloud plots: clean background vs adversarial background (test set) ---
    def subsample(arr, max_samples=2000):
        if arr.shape[0] > max_samples:
            idx = np.random.choice(arr.shape[0], max_samples, replace=False)
            return arr[idx]
        return arr

    clean_signal = subsample(X_test[y_test == 0], n_subsample)
    clean_background = subsample(bg_X_test_correct, n_subsample)
    adversarial_background = subsample(advs, n_subsample)

    plt.figure(figsize=(7, 6))
    plt.scatter(clean_background[:, 0], clean_background[:, 1], s=24, c='#B30326', alpha=0.5, label='Clean Background', edgecolor='black', linewidth=0.5)
    plt.scatter(clean_signal[:, 0], clean_signal[:, 1], s=24, c='#1f77b4', alpha=0.5, label='Clean Signal', edgecolor='black', linewidth=0.5)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "pointcloud_clean_test_signal_vs_background.pdf"), pad_inches=0.25, bbox_inches='tight', dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(adversarial_background[:, 0], adversarial_background[:, 1], s=24, c='#FF9900', alpha=0.5, label='Adversarial BG', edgecolor='black', linewidth=0.5)
    plt.scatter(clean_signal[:, 0], clean_signal[:, 1], s=24, c='#1f77b4', alpha=0.5, label='Clean Signal', edgecolor='black', linewidth=0.5)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "pointcloud_adversarial_vs_signal.pdf"), pad_inches=0.25, bbox_inches='tight', dpi=200)
    plt.close()

    # --- 1D histogram comparisons: clean background vs adversarial background (test set) ---
    for i, feat in enumerate(['x1', 'x2']):
        plt.figure(figsize=(8, 6))
        hist_clean, bin_edges = np.histogram(bg_X_test_correct[:, i], bins=50)
        plt.hist(bg_X_test_correct[:, i], bins=bin_edges, alpha=0.6, label='Clean Background', color='#1f77b4', density=True, histtype='stepfilled')
        plt.hist(advs[:, i], bins=bin_edges, alpha=0.6, label='Adversarial Background', color='#FF9900', density=True, histtype='stepfilled')
        plt.xlabel(feat)
        plt.ylabel('Density')
        plt.title(f'Clean vs Adversarial Background ({feat})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f"hist_clean_vs_adversarial_bg_{feat}.pdf"), bbox_inches='tight', dpi=150)
        plt.close()

        # Print the distributions and bin edges
        hist_adv, _ = np.histogram(advs[:, i], bins=bin_edges)
        print(f"\nFeature: {feat}")
        print(f"  Clean hist: {hist_clean.tolist()}")
        print(f"  Adv hist:   {hist_adv.tolist()}")
        print(f"  Bin edges:  {bin_edges.tolist()}")
        print(f"  N_clean: {bg_X_test_correct.shape[0]}, N_adv: {advs.shape[0]}")

        # Calculate JSD using the same method as in MultiGPUJSDCovAttack.py
        from scipy.spatial.distance import jensenshannon
        hist_clean = hist_clean + 1e-12
        hist_adv = hist_adv + 1e-12
        hist_clean = hist_clean / hist_clean.sum()
        hist_adv = hist_adv / hist_adv.sum()
        jsd = jensenshannon(hist_clean, hist_adv)
        print(f"  JSD: {jsd:.6f}")

    print(f"Fooling ratio (background misclassified as signal): {fooling_ratio:.4f}")


if __name__ == "__main__":
    main()