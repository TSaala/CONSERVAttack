import argparse
import sys
import numpy as np
import pandas as pd
import os
import keras
from keras.layers import Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import compute_class_weight
from scipy.spatial.distance import jensenshannon


# TODO: Fix paths etc. here, adjust numbers, run num etc.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Helpers import generate_adversarial_samples


def print_jsd_and_hist(X_clean, X_adv, label):
    for i, feat in enumerate(['x1', 'x2']):
        hist_clean, bin_edges = np.histogram(X_clean[:, i], bins=50, density=True)
        hist_adv, _ = np.histogram(X_adv[:, i], bins=bin_edges, density=True)
        hist_clean = (hist_clean + 1e-12) / (hist_clean + 1e-12).sum()
        hist_adv = (hist_adv + 1e-12) / (hist_adv + 1e-12).sum()
        jsd = jensenshannon(hist_clean, hist_adv)
        print(f"\n{label} - Feature {feat}:")
        print(f"  JSD: {jsd:.6f}")
        print(f"  Clean hist: {hist_clean.tolist()}")
        print(f"  Adv hist:   {hist_adv.tolist()}")
        print(f"  Bin edges:  {bin_edges.tolist()}")
        print(f"  N_clean: {X_clean.shape[0]}, N_adv: {X_adv.shape[0]}")


def get_correct_samples(X, y, model):
    y_pred = (model.predict(X) > 0.5).astype(int).flatten()
    correct_mask = (y_pred == y)
    return X[correct_mask], y[correct_mask]


def save_adversaries_df(advs, y_adv, fooled_indices, path):
    df = pd.DataFrame(advs, columns=['x1', 'x2'])
    df['Label'] = y_adv
    fooling = np.zeros(advs.shape[0], dtype=int)
    if fooled_indices is not None:
        fooling[fooled_indices] = 1
    df['fooling'] = fooling
    df.reset_index(drop=True, inplace=True)
    df.to_feather(path)


def load_adversaries_df(path):
    df = pd.read_feather(path)
    advs = df[['x1', 'x2']].to_numpy()
    y_adv = df['Label'].to_numpy()
    fooling = df['fooling'].to_numpy()
    return advs, y_adv, fooling


def generate_or_load_adversaries(X, y, model, model_file, attack_params, set_name, results_path, mask=None):
    adv_path = os.path.join(results_path, f"adversaries_{set_name}.feather")
    if os.path.exists(adv_path):
        print(f"Loading adversaries for {set_name} from checkpoint...")
        advs, y_adv, fooling = load_adversaries_df(adv_path)
        fooled_indices = np.where(fooling == 1)[0]
        return advs, y_adv, fooled_indices
    else:
        print(f"Generating adversaries for {set_name}...")
        advs = generate_adversarial_samples(
            x_test_correct=X,
            y_test_correct=y,
            model_weights_path=model_file,
            mask=mask,
            **attack_params
        )
        y_pred_adv = (model.predict(advs) > 0.5).astype(int).flatten()
        fooled_indices = np.where(y_pred_adv != y)[0]
        print(f"  {set_name}: {fooled_indices.shape[0]} adversarial samples (fooled)")
        save_adversaries_df(advs, y, fooled_indices, adv_path)
        return advs, y, fooled_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_prefix", type=str, required=True)
    parser.add_argument("--runcounter", type=int, required=True)
    parser.add_argument("--train_adv_size", type=int, required=True, default=5000)
    parser.add_argument("--val_adv_size", type=int, required=True, default=2000)
    parser.add_argument("--test_adv_size", type=int, required=True, default=2000)
    args = parser.parse_args()

    run = f'{args.run_prefix}_{args.runcounter}'

    path_to_data = './Data'
    results_path = f'./Results/{run}'
    model_path = f'./Models/{run}'

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # --- Load data ---
    df = pd.read_csv(os.path.join(path_to_data, 'donut_signal_background.csv'))
    X = df[['x1', 'x2']].to_numpy()
    y = df['Label'].to_numpy()  # 0: signal, 1: background

    # --- Train/val/test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # --- Background-only subsets ---
    X_train_bg = X_train[y_train == 1]
    y_train_bg = y_train[y_train == 1]
    X_val_bg = X_val[y_val == 1]
    y_val_bg = y_val[y_val == 1]
    X_test_bg = X_test[y_test == 1]
    y_test_bg = y_test[y_test == 1]

    # --- Train base classifier ---
    model_file = os.path.join(model_path, 'best_model.keras')
    if os.path.exists(model_file):
        model = keras.models.load_model(model_file)
        print(f"Loaded existing model from {model_file}")
    else:
        model = keras.Sequential([
            Dense(64, input_dim=2, activation='relu', kernel_regularizer=regularizers.L1(0.001)),
            BatchNormalization(),
            Dense(32, activation='relu', kernel_regularizer=regularizers.L1(0.001)),
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

    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # --- Get correctly classified background samples per split ---
    X_train_correct, y_train_correct = get_correct_samples(X_train_bg, y_train_bg, model)
    X_val_correct, y_val_correct = get_correct_samples(X_val_bg, y_val_bg, model)
    X_test_correct, y_test_correct = get_correct_samples(X_test_bg, y_test_bg, model)
    print(f"Correct train: {X_train_correct.shape[0]}, val: {X_val_correct.shape[0]}, test: {X_test_correct.shape[0]}")

    max_train = args.train_adv_size
    max_test = args.test_adv_size
    max_val = args.val_adv_size

    if X_train_correct.shape[0] > max_train:
        idx = np.random.choice(X_train_correct.shape[0], max_train, replace=False)
        X_train_correct = X_train_correct[idx]
        y_train_correct = y_train_correct[idx]
    if X_test_correct.shape[0] > max_test:
        idx = np.random.choice(X_test_correct.shape[0], max_test, replace=False)
        X_test_correct = X_test_correct[idx]
        y_test_correct = y_test_correct[idx]
    if X_val_correct.shape[0] > max_val:
        idx = np.random.choice(X_val_correct.shape[0], max_val, replace=False)
        X_val_correct = X_val_correct[idx]
        y_val_correct = y_val_correct[idx]

    # --- Generate adversarial samples ---
    attack_params = dict(
        min_change=0.001,
        step=0.001,
        n_iterations=10,
        num_bins=60,
        n_gpus=1,
        verbose=True,
        alpha=6.0,
        beta=1.0,
        use_no_change=True,
        max_jsd_single_change=0.01,
        max_frob_single_change=0.05,
        save_results=False,
        randomize_step=True,
        random_step_frac=0.5,
        num_candidates=150
    )

    mask_train = np.zeros(X_train_correct.shape[0], dtype=bool)
    mask_test = np.zeros(X_test_correct.shape[0], dtype=bool)
    mask_val = np.zeros(X_val_correct.shape[0], dtype=bool)

    advs_test, y_test_adv, test_fooled_indices = generate_or_load_adversaries(
        X_test_correct, y_test_correct, model, model_file, attack_params, "test", results_path, mask=mask_test)
    advs_train, y_train_adv, train_fooled_indices = generate_or_load_adversaries(
        X_train_correct, y_train_correct, model, model_file, attack_params, "train", results_path, mask=mask_train)
    advs_val, y_val_adv, val_fooled_indices = generate_or_load_adversaries(
        X_val_correct, y_val_correct, model, model_file, attack_params, "val", results_path, mask=mask_val)


    # --- Build detector training data ---
    X_train_clean = X_train
    y_train_clean = y_train
    if X_train_clean.shape[0] > max_train:
        idx = np.random.choice(X_train_clean.shape[0], max_train, replace=False)
        X_train_clean = X_train_clean[idx]
        y_train_clean = y_train_clean[idx]

    X_test_clean = X_test
    y_test_clean = y_test
    if X_test_clean.shape[0] > max_test:
        idx = np.random.choice(X_test_clean.shape[0], max_test, replace=False)
        X_test_clean = X_test_clean[idx]
        y_test_clean = y_test_clean[idx]

    X_val_clean = X_val
    y_val_clean = y_val
    if X_val_clean.shape[0] > max_val:
        idx = np.random.choice(X_val_clean.shape[0], max_val, replace=False)
        X_val_clean = X_val_clean[idx]
        y_val_clean = y_val_clean[idx]

    advs_train_fooled = advs_train[train_fooled_indices]
    advs_test_fooled = advs_test[test_fooled_indices]
    advs_val_fooled = advs_val[val_fooled_indices]
    X_train_combined = np.concatenate([X_train_clean, advs_train_fooled], axis=0)
    y_train_combined = np.concatenate([np.ones(X_train_clean.shape[0]), np.zeros(advs_train_fooled.shape[0])], axis=0)

    X_test_combined = np.concatenate([X_test_clean, advs_test_fooled], axis=0)
    y_test_combined = np.concatenate([np.ones(X_test_clean.shape[0]), np.zeros(advs_test_fooled.shape[0])], axis=0)

    X_val_combined = np.concatenate([X_val_clean, advs_val_fooled], axis=0)
    y_val_combined = np.concatenate([np.ones(X_val_clean.shape[0]), np.zeros(advs_val_fooled.shape[0])], axis=0)

    # --- Train adversarial detector ---
    classes = np.unique(y_train_combined)
    class_weight_dict = dict(zip(classes, compute_class_weight('balanced', classes=classes, y=y_train_combined)))

    detector = keras.Sequential([
        Dense(128, input_dim=2, activation='relu', kernel_regularizer=regularizers.L1(0.003)),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=regularizers.L1(0.003)),
        BatchNormalization(),
        Dense(32, activation='relu', kernel_regularizer=regularizers.L1(0.003)),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    detector.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    detector.fit(
        X_train_combined, y_train_combined,
        epochs=200,
        batch_size=256,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=2,
        class_weight=class_weight_dict,
        validation_data=(X_val_combined, y_val_combined)
    )

    detector_model_path = os.path.join(model_path, 'adversarial_detector_model.keras')
    detector.save(detector_model_path)
    print(f"Saved adversarial detector model to {detector_model_path}")

    score = detector.evaluate(X_test_combined, y_test_combined, verbose=1)
    print("Adversarial detector score (loss, accuracy):", score)

    # --- Misclassified clean indices ---
    y_pred_train_clean = np.round(detector.predict(X_train_correct).flatten())
    misclassified_train_indices = np.where(y_pred_train_clean != 1)[0]
    np.save(os.path.join(results_path, "misclassified_clean_indices_train.npy"), misclassified_train_indices)

    y_pred_test_clean = np.round(detector.predict(X_test_correct).flatten())
    misclassified_test_indices = np.where(y_pred_test_clean != 1)[0]
    np.save(os.path.join(results_path, "misclassified_clean_indices_test.npy"), misclassified_test_indices)

    print("Misclassified clean train indices:", misclassified_train_indices)
    print("Misclassified clean test indices:", misclassified_test_indices)

    # --- JSD diagnostics ---
    print("\n=== JSD & Histogram: All Clean vs All Adversarial (Test Set) ===")
    print_jsd_and_hist(X_test_correct, advs_test, "All Clean vs All Adv")

    if test_fooled_indices is not None and len(test_fooled_indices) > 0:
        print("\n=== JSD & Histogram: Fooled Clean vs Fooled Adv (Test Set) ===")
        print_jsd_and_hist(X_test_correct[test_fooled_indices], advs_test[test_fooled_indices], "Fooled Clean vs Fooled Adv")
        print(f"\nSanity check: N_clean_fooled = {X_test_correct[test_fooled_indices].shape[0]}, N_adv_fooled = {advs_test[test_fooled_indices].shape[0]}")
    else:
        print("\nNo fooled adversarial samples found in test set.")

    # --- Detector efficiency ---
    y_pred_signal = (detector.predict(X_test[y_test == 0]) > 0.5).astype(int).flatten()
    signal_efficiency = np.mean(y_pred_signal == 1)
    print(f"Detector efficiency on clean signal events: {signal_efficiency:.4f}")

    y_pred_adv = (detector.predict(advs_test_fooled) > 0.5).astype(int).flatten()
    adv_efficiency = np.mean(y_pred_adv == 0)
    print(f"Detector efficiency on adversarial samples (fooled): {adv_efficiency:.4f}")

    advs_not_detected = advs_test_fooled[y_pred_adv == 1]
    print(f"Number of clean signal events correctly classified by detector: {np.sum(y_pred_signal == 1)} of {X_test[y_test == 0].shape[0]}")
    print(f"Number of adversarial samples that fool and are NOT detected: {advs_not_detected.shape[0]} of {advs_test_fooled.shape[0]}")

    # --- Fooling ratios ---
    initial_fooling_ratio = len(test_fooled_indices) / X_test_correct.shape[0]
    corrected_fooling_ratio = advs_not_detected.shape[0] / X_test_correct.shape[0]
    print(f"Initial fooling ratio: {initial_fooling_ratio:.4f}")
    print(f"Corrected fooling ratio (fooled + not detected): {corrected_fooling_ratio:.4f}")

    # --- Save metrics ---
    metrics_path = os.path.join(results_path, "run_metrics.csv")
    metrics_df = pd.DataFrame([{
        "run_num": args.runcounter,
        "initial_fooling_ratio": initial_fooling_ratio,
        "corrected_fooling_ratio": corrected_fooling_ratio,
        "detector_efficiency_adv": adv_efficiency,
        "detector_efficiency_clean": signal_efficiency
    }])
    if os.path.exists(metrics_path):
        metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(metrics_path, mode='w', header=True, index=False)

    print("Done.")


if __name__ == "__main__":
    main()