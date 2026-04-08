import argparse
import sys
import os
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'legend.fontsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.linewidth': 2,
    'lines.linewidth': 2,
    'figure.figsize': (12, 9)
})
sns.set_style("whitegrid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_prefix", type=str, default="DonutDetector")
    parser.add_argument("--run_num", type=int, default=5)
    args = parser.parse_args()

    run_prefix = args.run_prefix
    run_range = range(args.run_num)

    base_model_dir = './Models'
    results_base_dir = './Results'
    data_path = './Data/donut_signal_background.csv'
    plots_dir = os.path.join(results_base_dir, "CrossDetector")
    os.makedirs(plots_dir, exist_ok=True)

    results = []

    # --- Pre-load total initially correct counts per adv run ---
    total_correct_dict = {}
    for adv_run in run_range:
        base_model_path = os.path.join(base_model_dir, f"{run_prefix}_{adv_run}", "best_model.keras")
        if os.path.isfile(base_model_path) and os.path.isfile(data_path):
            base_model = keras.models.load_model(base_model_path)
            df = pd.read_csv(data_path)
            X = df[['x1', 'x2']].to_numpy()
            y = df['Label'].to_numpy()
            y_pred = np.round(base_model.predict(X, verbose=0)).flatten()
            correct_mask = (y_pred == y.flatten())
            total_correct_dict[adv_run] = int(np.sum(correct_mask))
            print(f"Run {adv_run}: {total_correct_dict[adv_run]} correctly classified test samples.")
        else:
            total_correct_dict[adv_run] = None
            print(f"Run {adv_run}: base model or data not found, skipping total_correct.")

    # --- Cross-evaluation loop ---
    for detector_run in run_range:
        detector_model_path = os.path.join(
            base_model_dir, f"{run_prefix}_{detector_run}", "adversarial_detector_model.keras")
        if not os.path.isfile(detector_model_path):
            print(f"Detector model for run {detector_run} not found, skipping.")
            continue
        detector = keras.models.load_model(detector_model_path)
        print(f"\nLoaded detector from run {detector_run}.")

        # Detector efficiency on clean signal events
        if os.path.isfile(data_path):
            df = pd.read_csv(data_path)
            X = df[['x1', 'x2']].to_numpy()
            y = df['Label'].to_numpy()
            X_signal = X[y == 0]
            if len(X_signal) > 0:
                y_pred_clean = np.round(detector.predict(X_signal, verbose=0)).flatten()
                detector_efficiency_clean = float(np.mean(y_pred_clean == 1))
            else:
                detector_efficiency_clean = float('nan')
        else:
            detector_efficiency_clean = float('nan')

        for adv_run in run_range:
            adv_test_path = os.path.join(
                results_base_dir, f"{run_prefix}_{adv_run}", "adversaries_test.feather")
            if not os.path.isfile(adv_test_path):
                print(f"Adversarial test samples for run {adv_run} not found, skipping.")
                continue

            adv_df = pd.read_feather(adv_test_path)
            X_adv = adv_df[['x1', 'x2']].to_numpy()
            y_adv = adv_df['Label'].to_numpy()
            num_adv_samples = len(X_adv)

            y_pred_adv = np.round(detector.predict(X_adv, verbose=0)).flatten()
            num_adv_detected = int(np.sum(y_pred_adv == 0))
            num_adv_not_detected = int(np.sum(y_pred_adv == 1))
            fraction_adv_correctly_classified = (
                num_adv_detected / num_adv_samples if num_adv_samples > 0 else float('nan')
            )

            base_model_path = os.path.join(
                base_model_dir, f"{run_prefix}_{detector_run}", "best_model.keras")
            if os.path.isfile(base_model_path):
                base_model = keras.models.load_model(base_model_path)
                y_base_pred = np.round(base_model.predict(X_adv, verbose=0)).flatten()
                initial_fooling_ratio = float(np.mean(y_base_pred != y_adv.flatten()))
            else:
                initial_fooling_ratio = float('nan')

            corrected_fooling_ratio = initial_fooling_ratio * (1 - fraction_adv_correctly_classified)
            run_type = "SAME" if adv_run == detector_run else "CROSS"
            total_initially_correct = total_correct_dict.get(adv_run, None)

            print(
                f"Detector {detector_run} on adversaries from run {adv_run} [{run_type}]: "
                f"initial_fooling_ratio={initial_fooling_ratio:.4f}, "
                f"corrected_fooling_ratio={corrected_fooling_ratio:.4f}, "
                f"fraction_adv_correctly_classified={fraction_adv_correctly_classified:.4f}, "
                f"detector_efficiency_clean={detector_efficiency_clean:.4f}"
            )

            results.append({
                "detector_run": detector_run,
                "adv_run": adv_run,
                "run_type": run_type,
                "initial_fooling_ratio": initial_fooling_ratio,
                "corrected_fooling_ratio": corrected_fooling_ratio,
                "fraction_adv_correctly_classified": fraction_adv_correctly_classified,
                "num_samples": num_adv_samples,
                "total_initially_correct": total_initially_correct,
                "detector_efficiency_clean": detector_efficiency_clean
            })

    if not results:
        print("No results collected. Check that model and adversary files exist.")
        return

    results_df = pd.DataFrame(results)

    # --- Summary statistics ---
    same_run = results_df[results_df["run_type"] == "SAME"]
    cross_run = results_df[results_df["run_type"] == "CROSS"]

    print("\n--- SAME-RUN STATISTICS ---")
    print(f"Initial fooling ratio:               {same_run['initial_fooling_ratio'].mean():.4f}")
    print(f"Fraction adv correctly classified:   {same_run['fraction_adv_correctly_classified'].mean():.4f}")
    print(f"Detector efficiency on clean events: {same_run['detector_efficiency_clean'].mean():.4f}")
    print(f"Corrected fooling ratio:             {same_run['corrected_fooling_ratio'].mean():.4f}")

    print("\n--- CROSS-RUN STATISTICS ---")
    print(f"Average initial fooling ratio:               {cross_run['initial_fooling_ratio'].mean():.4f}")
    print(f"Average fraction adv correctly classified:   {cross_run['fraction_adv_correctly_classified'].mean():.4f}")
    print(f"Average detector efficiency on clean events: {cross_run['detector_efficiency_clean'].mean():.4f}")
    print(f"Average corrected fooling ratio:             {cross_run['corrected_fooling_ratio'].mean():.4f}")

    # --- Plots (cross runs only) ---
    grouped = cross_run.groupby('detector_run')

    # 1. Average detector efficiency on adversaries per detector run
    avg_robustness = grouped['fraction_adv_correctly_classified'].mean()
    std_robustness = grouped['fraction_adv_correctly_classified'].std()

    fig, ax = plt.subplots(figsize=(12.0, 9.0))
    cap_width = 0.35
    for i, idx in enumerate(avg_robustness.index):
        mean = avg_robustness.values[i]
        std = std_robustness.values[i]
        ax.errorbar(idx, mean, yerr=std, linestyle='', fmt='o', capsize=0, capthick=4,
                    markersize=12, c='#B30326', alpha=1.0, zorder=1, elinewidth=4)
        ax.plot([idx - cap_width, idx + cap_width], [mean + std, mean + std], color='#B30326', lw=2, zorder=1)
        ax.plot([idx - cap_width, idx + cap_width], [mean - std, mean - std], color='#B30326', lw=2, zorder=1)
        ax.scatter(idx, mean, c='#B30326', s=200, marker='o', edgecolor='black', linewidth=2, zorder=2)
    ax.set_ylabel('Detector Efficiency on Adversaries', fontsize=24)
    ax.set_xlabel('Detector Run', fontsize=24)
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "avg_robustness_per_detector_cross.pdf"),
                pad_inches=0.25, bbox_inches='tight', dpi=200)
    plt.close()

    # 2. Initial and corrected fooling ratios per detector run
    mean_initial = grouped['initial_fooling_ratio'].mean()
    std_initial = grouped['initial_fooling_ratio'].std()
    mean_corrected = grouped['corrected_fooling_ratio'].mean()
    std_corrected = grouped['corrected_fooling_ratio'].std()

    fig, ax = plt.subplots(figsize=(12.0, 9.0))
    for i, idx in enumerate(mean_initial.index):
        mean = mean_initial.values[i]
        std = std_initial.values[i]
        ax.errorbar(idx - 0.1, mean, yerr=std, linestyle='', fmt='o', capsize=0, capthick=4,
                    markersize=12, c='#B30326', alpha=1.0, zorder=1, elinewidth=4,
                    label='Initial Fooling Ratio' if i == 0 else None)
        ax.plot([idx - 0.1 - cap_width, idx - 0.1 + cap_width], [mean + std, mean + std], color='#B30326', lw=2, zorder=1)
        ax.plot([idx - 0.1 - cap_width, idx - 0.1 + cap_width], [mean - std, mean - std], color='#B30326', lw=2, zorder=1)
        ax.scatter(idx - 0.1, mean, c='#B30326', s=200, marker='o', edgecolor='black', linewidth=2, zorder=2)
    for i, idx in enumerate(mean_corrected.index):
        mean = mean_corrected.values[i]
        std = std_corrected.values[i]
        ax.errorbar(idx + 0.1, mean, yerr=std, linestyle='', fmt='o', capsize=0, capthick=4,
                    markersize=12, c='dimgray', alpha=1.0, zorder=1, elinewidth=4,
                    label='Corrected Fooling Ratio' if i == 0 else None)
        ax.plot([idx + 0.1 - cap_width, idx + 0.1 + cap_width], [mean + std, mean + std], color='dimgray', lw=2, zorder=1)
        ax.plot([idx + 0.1 - cap_width, idx + 0.1 + cap_width], [mean - std, mean - std], color='dimgray', lw=2, zorder=1)
        ax.scatter(idx + 0.1, mean, c='dimgray', s=200, marker='o', edgecolor='black', linewidth=2, zorder=2)
    ax.set_xlabel('Detector Run', fontsize=24)
    ax.set_ylabel('Fooling Ratio', fontsize=24)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "avg_fooling_ratios_scatter_cross.pdf"),
                pad_inches=0.25, bbox_inches='tight', dpi=200)
    plt.close()

    # 3. TPR_adv vs predicted corrected fooling ratio curve
    avg_init_fr = grouped['initial_fooling_ratio'].mean().mean()
    tpr_adv_curve = np.linspace(0, 1, 10)
    pred_corr_fr_curve = avg_init_fr * (1 - tpr_adv_curve)

    fig, ax = plt.subplots(figsize=(12.0, 9.0))
    ax.plot(tpr_adv_curve, pred_corr_fr_curve, color='#B30326', linewidth=3,
            marker='o', markersize=12, markeredgecolor='black', markerfacecolor='#B30326',
            alpha=1.0, zorder=2)
    ax.set_xlabel('Detector Classification Efficiency', fontsize=24)
    ax.set_ylabel('Predicted Corrected Fooling Ratio', fontsize=24)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(pred_corr_fr_curve.max(), 0.1) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "tpr_vs_predicted_corr_fooling_curve_cross.pdf"),
                pad_inches=0.25, bbox_inches='tight', dpi=200)
    plt.close()

    print(f"\nSaved plots to {plots_dir}")


if __name__ == "__main__":
    main()