import numpy as np
import multiprocessing
from cupyx import scatter_add
import matplotlib.pyplot as plt
import gc
import time
import os
import cupy as cp
from collections import Counter
from scipy.spatial.distance import jensenshannon

multiprocessing.set_start_method('spawn', force=True)
import tensorflow as tf


def precompute_bins_for_all_features(x_test_correct, num_bins=100):
    bin_edges_dict = {}
    min_values = []
    max_values = []
    for feature in range(x_test_correct.shape[1]):
        feature_data = x_test_correct[:, feature]
        min_val, max_val = np.min(feature_data), np.max(feature_data)
        min_values.append(min_val)
        max_values.append(max_val)
        bin_edges_dict[feature] = np.linspace(min_val, max_val, num_bins + 1)
    return bin_edges_dict, np.array(min_values), np.array(max_values)



def global_histogram_gpu(datasets, bin_edges_dict):
    datasets_gpu = cp.asarray(datasets, dtype=cp.float32)
    # Use the first key in bin_edges_dict
    first_key = next(iter(bin_edges_dict))
    num_bins = bin_edges_dict[first_key].shape[0] - 1
    global_hist_vals = cp.empty((num_bins, len(bin_edges_dict)), dtype=cp.float32)
    for i, feature in enumerate(bin_edges_dict):
        bin_edges = cp.asarray(bin_edges_dict[feature])
        hist_vals, _ = cp.histogram(datasets_gpu[:, feature], bins=bin_edges, density=True)
        global_hist_vals[:, i] = hist_vals
    cp.cuda.Device(0).synchronize()
    return global_hist_vals



def compute_js_distance_histogram_gpu(original_hist, modified_hist, bin_edges):
    eps = 1e-10
    # Replace small values with eps
    original_hist = cp.maximum(original_hist, eps)
    modified_hist = cp.maximum(modified_hist, eps)
    # Normalize to sum to 1
    original_hist = original_hist / cp.sum(original_hist)
    modified_hist = modified_hist / cp.sum(modified_hist)
    M = 0.5 * (original_hist + modified_hist)
    M_safe = cp.maximum(M, eps)
    js_divergence = 0.5 * (cp.sum(original_hist * cp.log2(original_hist / M_safe)) +
                           cp.sum(modified_hist * cp.log2(modified_hist / M_safe)))
    js_divergence = cp.maximum(js_divergence, 0.0)
    js_distance = cp.sqrt(js_divergence)
    return js_distance.item()



def compute_js_distance_feature_gpu_batch(original_hists, candidate_hists):
    eps = 1e-10
    original_hists = cp.maximum(original_hists, eps)
    candidate_hists = cp.maximum(candidate_hists, eps)
    original_hists = original_hists / cp.sum(original_hists, axis=1, keepdims=True)
    candidate_hists = candidate_hists / cp.sum(candidate_hists, axis=2, keepdims=True)
    original_hists_exp = original_hists[:, None, :]  # (num_features, 1, num_bins)
    M = 0.5 * (original_hists_exp + candidate_hists)
    M = cp.maximum(M, eps)  # Only replace zeros in M for safety
    js_divergence = 0.5 * (
        cp.sum(candidate_hists * cp.log2(candidate_hists / M), axis=2) +
        cp.sum(original_hists_exp * cp.log2(original_hists_exp / M), axis=2)
    )
    js_distance = cp.sqrt(js_divergence)
    return js_distance



def get_correlation_matrix(X):
    if isinstance(X, np.ndarray):
        return np.corrcoef(X, rowvar=False)
    else:
        return cp.corrcoef(X, rowvar=False)
    


def optimize_fooled_samples(
    model, x_test_modified, y_test_correct, mask, min_values, max_values, bin_edges_dict,
    feature_indices, step=0.01, max_search_steps=10
):
    """
    For samples already fooled (mask == True), search for a nearby value that still fools the model
    and yields a lower JSD/FN than the current adversarial value.
    """
    import tensorflow as tf
    import cupy as cp

    for sample_idx in np.where(mask)[0]:
        for feature_idx in feature_indices:
            current_value = x_test_modified[sample_idx, feature_idx]
            # Search in a small window around current value
            min_val = min_values[feature_idx]
            max_val = max_values[feature_idx]
            search_range = np.linspace(
                max(min_val, current_value - step * max_search_steps),
                min(max_val, current_value + step * max_search_steps),
                num=2 * max_search_steps + 1
            )
            best_jsd = None
            best_fn = None
            best_value = current_value
            # Compute current JSD/FN
            orig_hist = cp.asarray(np.histogram(x_test_modified[:, feature_idx], bins=bin_edges_dict[feature_idx], density=True)[0])
            for candidate in search_range:
                x_candidate = x_test_modified.copy()
                x_candidate[sample_idx, feature_idx] = candidate
                # Check if still fooled
                pred = model.predict(x_candidate[sample_idx:sample_idx+1], verbose=False)
                if pred.shape[-1] == 1:
                    pred = pred.flatten()
                pred_class = (pred > 0.5).astype(int)
                if pred_class[0] != y_test_correct[sample_idx]:
                    # Compute JSD
                    cand_hist = cp.asarray(np.histogram(x_candidate[:, feature_idx], bins=bin_edges_dict[feature_idx], density=True)[0])
                    jsd = compute_js_distance_histogram_gpu(orig_hist, cand_hist, cp.asarray(bin_edges_dict[feature_idx]))
                    # Compute FN (Frobenius norm of correlation diff)
                    corr_clean = get_correlation_matrix(x_test_modified)
                    corr_adv = get_correlation_matrix(x_candidate)
                    frob_corr_clean = np.linalg.norm(corr_clean, ord='fro')
                    frob_corr_diff = np.linalg.norm(corr_adv - corr_clean, ord='fro')
                    fn = frob_corr_diff / (frob_corr_clean + 1e-10)
                    # Accept only if JSD/FN is strictly lower than current
                    if (best_jsd is None or jsd < best_jsd) and (best_fn is None or fn < best_fn):
                        best_jsd = jsd
                        best_fn = fn
                        best_value = candidate
            # Only update if improvement found
            if best_value != current_value:
                x_test_modified[sample_idx, feature_idx] = best_value
    return x_test_modified



def update_histogram_incremental_batch_batch(hists_before, bin_edges_batch, new_values_batch, old_values_batch, bs):
    """
    Vectorized histogram update for all features and all candidates.
    hists_before: (num_features, num_bins)
    bin_edges_batch: (num_features, num_bins+1)
    new_values_batch, old_values_batch: (num_features, num_candidates)
    Returns: (num_features, num_candidates, num_bins)
    """
    num_features, num_bins = hists_before.shape
    num_candidates = new_values_batch.shape[1]

    # Clip new/old values to bin range
    new_values = cp.clip(new_values_batch, bin_edges_batch[:, 0:1], bin_edges_batch[:, -1:] - 1e-6)
    old_values = cp.clip(old_values_batch, bin_edges_batch[:, 0:1], bin_edges_batch[:, -1:] - 1e-6)

    # Fully vectorized searchsorted using broadcasting+sum
    old_bins = cp.sum(old_values[..., None] >= bin_edges_batch[:, None, :], axis=2) - 1
    new_bins = cp.sum(new_values[..., None] >= bin_edges_batch[:, None, :], axis=2) - 1
    old_bins = cp.clip(old_bins, 0, num_bins - 1)
    new_bins = cp.clip(new_bins, 0, num_bins - 1)

    # Prepare histograms
    bin_width = bin_edges_batch[:, 1] - bin_edges_batch[:, 0]
    hist_raw = cp.round(hists_before * bs * bin_width[:, None]).astype(cp.int32)
    histograms_bin_shift = cp.tile(hist_raw[:, None, :], (1, num_candidates, 1)).astype(cp.int32)

    # Prepare indices for scatter_add
    feature_idx = cp.arange(num_features)[:, None]
    candidate_idx = cp.arange(num_candidates)[None, :]
    scatter_add(histograms_bin_shift, (feature_idx, candidate_idx, old_bins), -cp.ones_like(old_bins, dtype=cp.int32))
    scatter_add(histograms_bin_shift, (feature_idx, candidate_idx, new_bins), cp.ones_like(new_bins, dtype=cp.int32))

    histograms_bin_shift = histograms_bin_shift / (bs * bin_width[:, None, None])
    return histograms_bin_shift



def batch_correlation_update_single_change_vectorized(
    X_cp, corr_clean_cp, cov_clean_cp, mean_clean_cp, var_clean_cp, index, new_values
):
    """
    Vectorized computation of Frobenius norms of correlation matrix differences for all features and all candidates,
    only for the affected row/column.
    Returns: (num_features, num_candidates) Frobenius norms
    """
    n, d = X_cp.shape
    num_features, num_candidates = new_values.shape

    
    # --- Pearson logic (unchanged) ---
    x_old = X_cp[index, :]  # (d,)
    mean_new = mean_clean_cp[:, None] + (new_values - x_old[:, None]) / n  # (d, num_candidates)
    var_new = var_clean_cp[:, None] + (
        (new_values - mean_new) ** 2 - (x_old[:, None] - mean_clean_cp[:, None]) ** 2
    ) / (n - 1)  # (d, num_candidates)

    diff_old = X_cp[index, :] - mean_clean_cp  # (d,)
    diff_new = cp.broadcast_to(diff_old[:, None], (d, num_candidates)).copy()  # (d, num_candidates)
    diff_new[cp.arange(d), :] = new_values - mean_new  # update diagonal

    # Compute new covariance rows for each feature/candidate
    diff_new_T = diff_new.transpose(1, 0)  # (num_candidates, d)
    outer = diff_new_T[:, :, None] * diff_new_T[:, None, :]  # (num_candidates, d, d)
    outer = outer.transpose(1, 0, 2)  # (d, num_candidates, d)

    diff_old_outer = diff_old[:, None] * diff_old[None, :]  # (d, d)
    diff_old_outer = cp.tile(diff_old_outer[:, None, :], (1, num_candidates, 1))  # (d, num_candidates, d)

    cov_new_j = cov_clean_cp[cp.arange(d), :][:, None, :] + (
        outer - diff_old_outer
    ) / (n - 1)

    # Update variance vector for each candidate
    std_new = cp.sqrt(var_new + 1e-10)  # (d, num_candidates)
    std_clean = cp.sqrt(var_clean_cp + 1e-10)  # (d,)

    # Compute new correlation rows for each feature/candidate
    denom = std_new[:, :, None] * std_clean[None, None, :]  # (d, num_candidates, d)
    corr_new_j = cov_new_j / (denom + 1e-10)

    # Compute Frobenius norm of the difference for only the affected row and column
    row_diff = corr_new_j - corr_clean_cp[cp.arange(d), None, :]  # (d, num_candidates, d)
    col_diff = corr_new_j.transpose(2, 1, 0) - corr_clean_cp[:, None, cp.arange(d)]  # (d, num_candidates, d)
    frob_sq = cp.sum(row_diff ** 2, axis=2) + cp.sum(col_diff ** 2, axis=2)
    diag_diff = corr_new_j[cp.arange(d), :, cp.arange(d)] - corr_clean_cp[cp.arange(d), cp.arange(d)][:, None]
    frob_sq -= diag_diff ** 2
    frob = cp.sqrt(frob_sq)
    return frob



def find_best_changes_for_sample_vectorized(
    index, original_hist_vals, modified_values, gradients,
    min_values, max_values, bin_edges_dict, bs, min_change=0.1, step=0.01, feature_idx_chunk=None,
    alpha=1.0, beta=1.0, max_jsd_single_change=0.005, max_frob_single_change=0.005, use_no_change=False,
    randomize_steps=False, random_step_frac=0.2, num_candidates=None
):
    """
    Fully vectorized: For a given sample, find the best change for each feature that minimally disturbs the feature distribution.
    Uses batch histogram and JS distance computation over both features and candidates.
    """
    num_features = modified_values.shape[1]
    original_values = modified_values[index]
    best_changes = cp.zeros(num_features, dtype=cp.float32)


    # 1. Generate candidate changes for each feature
    candidate_changes_list = []
    max_candidates = 0
    for feature in range(num_features):
        gradient = gradients[feature]
        min_value, max_value = min_values[feature], max_values[feature]
        change_direction = 1 if gradient >= 0 else -1

        if randomize_steps:
            step_rand = step * np.random.uniform(1 - random_step_frac, 1 + random_step_frac)
            min_change_rand = min_change * np.random.uniform(1 - random_step_frac, 1 + random_step_frac)
        else:
            step_rand = step
            min_change_rand = min_change
        
        if num_candidates != None:
            step_rand = step
            min_change_rand = min_change
            possible_changes = cp.random.uniform(min_change_rand, max_value - original_values[feature], size=num_candidates)
        elif change_direction == 1:
            possible_changes = cp.arange(min_change_rand, max_value - original_values[feature] + step_rand, step_rand)
        else:
            possible_changes = cp.arange(min_change_rand, original_values[feature] - min_value + step_rand, step_rand)
        possible_changes = possible_changes * change_direction
        # Add "no change" candidate (0.0) at the start
        if use_no_change:
            possible_changes = cp.concatenate([cp.array([0.0]), possible_changes])
        candidate_changes_list.append(possible_changes)
        max_candidates = max(max_candidates, possible_changes.size)

    # 2. Pad candidate changes and create mask
    candidate_changes_padded = cp.zeros((num_features, max_candidates), dtype=cp.float32)
    mask = cp.zeros((num_features, max_candidates), dtype=cp.bool_)
    for feature in range(num_features):
        n = candidate_changes_list[feature].size
        if n > 0:
            candidate_changes_padded[feature, :n] = candidate_changes_list[feature]
            mask[feature, :n] = True

    if max_candidates == 0:
        # Return zeros or some default, and skip further computation
        best_changes = cp.zeros(num_features, dtype=cp.float32)
        best_idx = cp.zeros(num_features, dtype=cp.int32)
        return index, best_changes, best_idx.get()

    # 3. Prepare new_values and old_values arrays
    new_values = original_values[:, None] + candidate_changes_padded
    new_values = cp.clip(new_values, cp.asarray(min_values)[:, None], cp.asarray(max_values)[:, None])
    old_values = cp.broadcast_to(original_values[:, None], new_values.shape)

    # 4. Prepare current histograms and bin edges for all features
    current_hists = cp.stack([
        cp.histogram(modified_values[:, i], bins=bin_edges_dict[global_feature], density=True)[0]
        for i, global_feature in enumerate(feature_idx_chunk)
    ])
    bin_edges_batch = cp.stack([cp.asarray(bin_edges_dict[global_feature]) for global_feature in feature_idx_chunk])

    batch_histograms = update_histogram_incremental_batch_batch(
        current_hists, bin_edges_batch, new_values, old_values, bs
    )

    js_distances = compute_js_distance_feature_gpu_batch(
        original_hist_vals.T, batch_histograms
    )
    js_distances = cp.where(mask, js_distances, cp.inf)

    del batch_histograms
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    # 7. Vectorized correlation update and Frobenius norm for all features/candidates
    corr_clean_cp = cp.asarray(get_correlation_matrix(modified_values))
    cov_clean_cp = cp.asarray(cp.cov(modified_values, rowvar=False))
    mean_clean_cp = cp.asarray(cp.mean(modified_values, axis=0))
    var_clean_cp = cp.asarray(cp.var(modified_values, axis=0))

    corr_distances = batch_correlation_update_single_change_vectorized(
        cp.asarray(modified_values), corr_clean_cp, cov_clean_cp, mean_clean_cp, var_clean_cp, index, new_values)
    corr_distances = cp.where(mask, corr_distances, cp.inf)
    
    # --- Per-feature constraints ---
    valid_mask = (
        (js_distances <= max_jsd_single_change) &
        (corr_distances <= max_frob_single_change)
    )
    total_cost = alpha * js_distances + beta * corr_distances

    best_idx = cp.zeros(num_features, dtype=cp.int32)
    for feature in range(num_features):
        # Only consider candidates (excluding "no change")
        if use_no_change:
            valid_candidates = cp.where(valid_mask[feature, 1:])[0] + 1
            if valid_candidates.size > 0:
                costs = total_cost[feature, valid_candidates]
                best_valid = valid_candidates[cp.argmin(costs)]
                best_idx[feature] = best_valid
            else:
                best_idx[feature] = 0  # "no change"
        else:
            all_candidates = cp.where(valid_mask[feature])[0]
            if all_candidates.size > 0:
                costs = total_cost[feature, all_candidates]
                best_valid = all_candidates[cp.argmin(costs)]
                best_idx[feature] = best_valid
            else:
                best_idx[feature] = 0  # fallback to "no change" if none valid

    best_changes = candidate_changes_padded[cp.arange(num_features), best_idx]
    del js_distances

    return index, best_changes, best_idx.get()



def load_model_generic(model_weights_path):
    if model_weights_path.endswith('.hdf5') or model_weights_path.endswith('.h5') or model_weights_path.endswith('.keras'):
        import keras
        model = keras.models.load_model(model_weights_path)
        return model, 'keras'
    elif model_weights_path.endswith('.pth'):
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(model_weights_path, map_location=device)
        model.eval()
        return model, 'torch'
    else:
        raise ValueError(f"Unknown model file type: {model_weights_path}")
    


def adversarial_modify_features_on_gpu(
    gpu_idx, feature_indices, x_test_correct, x_test_modified, y_test_correct, mask,
    min_values, max_values, bin_edges_dict, model_weights_path,
    min_change, step, alpha=1.0, beta=1.0, 
    use_no_change=False, max_jsd_single_change=0.005, max_frob_single_change=0.005,
    model_type=None, randomize_steps=False, random_step_frac=0.2,
    num_candidates=None
):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    cp.cuda.Device(0).use()

    # Load model according to type
    if model_type is None:
        model, model_type = load_model_generic(model_weights_path)
    else:
        model, _ = load_model_generic(model_weights_path)

    bin_edges_dict_chunk = {k: v for k, v in bin_edges_dict.items() if k in feature_indices}
    original_hist_vals_chunk = global_histogram_gpu(x_test_correct, bin_edges_dict_chunk)
    feature_idx_chunk = feature_indices
        
    valid_indices, updated_chunk, best_candidate_indices = apply_changes_to_batch(
        model,
        x_test_modified,
        y_test_correct,
        mask,
        min_values,
        max_values,
        bin_edges_dict_chunk,
        original_hist_vals_chunk,
        feature_idx_chunk,
        min_change,
        step,
        alpha=alpha,
        beta=beta,
        use_no_change=use_no_change,
        max_jsd_single_change=max_jsd_single_change,
        max_frob_single_change=max_frob_single_change,
        randomize_steps=randomize_steps,
        random_step_frac=random_step_frac,
        num_candidates=num_candidates
    )
    return feature_idx_chunk, valid_indices, updated_chunk, best_candidate_indices



def apply_changes_to_batch(
    model, x_test_modified, y_batch, mask, min_values, max_values,
    bin_edges_dict_chunk, original_hist_vals_chunk, feature_idx_chunk, min_change=0.01, step=0.01,
    minibatch_size=512, alpha=1.0, beta=1.0, 
    use_no_change=False, max_jsd_single_change=0.005, max_frob_single_change=0.005,
    randomize_steps=False, random_step_frac=0.2, num_candidates=None
):
    from tqdm import tqdm 

    modified_batch = cp.asarray(x_test_modified)
    min_values_cp = cp.asarray(min_values)
    max_values_cp = cp.asarray(max_values)
    x_batch_tensor = tf.convert_to_tensor(x_test_modified, dtype=tf.float32)
    original_hist_vals_chunk = cp.asarray(original_hist_vals_chunk)
    valid_indices = np.where(~mask)[0]
    bs = x_test_modified.shape[0]

    for batch_num, start in tqdm(enumerate(range(0, len(valid_indices), minibatch_size)), desc="Mini-batches", leave=False):
        end = min(start + minibatch_size, len(valid_indices))
        batch_indices = valid_indices[start:end]
        #x_valid = x_test_modified[batch_indices]
        #y_valid = y_batch[batch_indices]

        x_valid = tf.gather(x_batch_tensor, batch_indices)
        y_valid = tf.gather(y_batch, batch_indices)

        # Compute gradients for this mini-batch
        with tf.GradientTape() as tape:
            tape.watch(x_valid)
            predictions = model(x_valid)
            predictions = tf.squeeze(predictions, axis=-1)
            loss = tf.keras.losses.binary_crossentropy(y_valid, predictions)
        grads = tape.gradient(loss, x_valid).numpy()  # shape: (mini_bs, n_features)

        best_candidate_indices = {}

        for i, idx in enumerate(batch_indices):
            _, best_changes, best_idx = find_best_changes_for_sample_vectorized(
                idx,
                original_hist_vals_chunk,
                modified_batch[:, feature_idx_chunk],
                grads[i][feature_idx_chunk],
                min_values[feature_idx_chunk],
                max_values[feature_idx_chunk],
                {k: bin_edges_dict_chunk[k] for k in feature_idx_chunk},
                bs,
                min_change,
                step,
                feature_idx_chunk,
                alpha=alpha,
                beta=beta,
                max_jsd_single_change=max_jsd_single_change,
                max_frob_single_change=max_frob_single_change,
                use_no_change=use_no_change,
                randomize_steps=randomize_steps,
                random_step_frac=random_step_frac,
                num_candidates=num_candidates
            )
            modified_batch[idx, feature_idx_chunk] = cp.clip(
                modified_batch[idx, feature_idx_chunk] + best_changes,
                min_values_cp[feature_idx_chunk],
                max_values_cp[feature_idx_chunk]
            )

            for j, feat in enumerate(feature_idx_chunk):
                if feat not in best_candidate_indices:
                    best_candidate_indices[feat] = []
                best_candidate_indices[feat].append(best_idx[j])

    return valid_indices, modified_batch[valid_indices[:, None], feature_idx_chunk].get(), best_candidate_indices



# TODO: Fix this, remove parameters we do need and debug approaches we dont need here
def generate_adversarial_samples(
    x_test_correct,
    y_test_correct,
    model_weights_path,
    min_change=0.003,
    step=0.01,
    n_iterations=10,
    n_gpus=4,
    mask=None,
    num_bins=100,
    verbose=True,
    alpha=1.0,
    beta=1.0,
    cluster=None,
    save_dir=None,
    max_jsd_single_change=0.005,
    max_frob_single_change=0.005,
    use_no_change=False,
    save_results=True,
    track_best_constraint_iter=False,
    constraint_jsd=0.06,
    constraint_rel_corr=0.006,
    optimize_already_fooled=False,
    randomize_step=False,
    random_step_frac=0.2,
    num_candidates=None
):
    """
    Generates adversarial samples using the multi-GPU vectorized approach.
    Returns: x_test_modified (adversarial samples)
    """
    n_features = x_test_correct.shape[1]
    model, model_type = load_model_generic(model_weights_path)
    # --- Only split features if more than 2 ---
    if n_features > 2:
        feature_indices = np.array_split(np.arange(n_features), n_gpus)
    else:
        feature_indices = [np.arange(n_features)]
    if mask is None:
        mask = np.zeros(x_test_correct.shape[0], dtype=bool)
    x_test_modified = x_test_correct.copy()
    bin_edges_dict, min_values, max_values = precompute_bins_for_all_features(x_test_correct, num_bins=num_bins)

    fooling_ratios = []
    js_distances_per_iteration = []
    frob_corr_diffs_per_iteration = []

    n_samples = x_test_correct.shape[0]

    if save_dir is None:
        if cluster == 'Physik':
            save_dir = f'/cephfs/user/tsaala/TopoDNN/NewAttack/Models/TestJSDCovAttack'
        else:
            save_dir = f'/home/tsaala_hpc/results/TopoDNN/TestJSDCorrAttackNewPlots'
    os.makedirs(save_dir, exist_ok=True)

    best_constraint_iter = None
    best_constraint_fooling = -1
    best_constraint_jsd = None
    best_constraint_rel_corr = None

    for iteration in range(n_iterations):
        iter_start = time.time()
        iter_save_dir = os.path.join(save_dir, f"Iteration_{iteration+1}")
        os.makedirs(iter_save_dir, exist_ok=True)
        if verbose:
            print(f"Iteration {iteration + 1}/{n_iterations}")

        random_feature_idx = np.random.randint(0, x_test_modified.shape[1])

        n_features = x_test_correct.shape[1]
        if n_features > 2:
            feature_indices = np.array_split(np.arange(n_features), n_gpus)
        else:
            feature_indices = [np.arange(n_features)]

        args = [
            (
                i,
                feature_indices[i],
                np.asarray(x_test_correct),
                np.asarray(x_test_modified),
                np.asarray(y_test_correct),
                np.asarray(mask),
                min_values,
                max_values,
                bin_edges_dict,
                model_weights_path,
                min_change,
                step,
                alpha,
                beta,
                use_no_change,
                max_jsd_single_change,
                max_frob_single_change,
                model_type,
                randomize_step,
                random_step_frac,
                num_candidates
            )
            for i in range(len(feature_indices))
        ]
        with multiprocessing.get_context("spawn").Pool(n_gpus) as pool:
            results = pool.starmap(adversarial_modify_features_on_gpu, args)

        # --- Track candidate indices for the random feature per iteration ---
        feature_counts = Counter()
        total_samples_considered = 0
        for feature_idx_chunk, valid_indices, modified_chunk, best_candidate_indices in results:
            x_test_modified[valid_indices[:, None], feature_idx_chunk] = modified_chunk
            if random_feature_idx in feature_idx_chunk:
                candidate_list = best_candidate_indices[random_feature_idx]
                candidate_indices = candidate_list
                feature_counts.update(candidate_indices)
                total_samples_considered += len(candidate_indices)

        # --- Reconstruct candidate changes for the random feature after each iteration ---
        f = random_feature_idx
        delta = x_test_modified[:, f] - x_test_correct[:, f]
        abs_delta = np.abs(delta)
        valid = abs_delta >= min_change
        candidate_idx = np.round((abs_delta - min_change) / step).astype(int)
        candidate_idx = candidate_idx[valid]
        candidate_counts = Counter(candidate_idx)
        total_samples_considered = len(candidate_idx)

        if save_results:
            counts_path = os.path.join(save_dir, f"feature{random_feature_idx}_candidate_counts_per_iter.txt")
            with open(counts_path, "a") as f:
                if iteration == 0:
                    f.write(f"Candidate choice counts for feature {random_feature_idx} per iteration:\n")
                f.write(f"Iteration {iteration+1}:\n")
                for cand_idx, count in sorted(candidate_counts.items()):
                    f.write(f"  Candidate {cand_idx}: {count} samples\n")
                f.write(f"Total samples considered for feature {random_feature_idx}: {total_samples_considered}\n\n")

            print(f"Candidate choice counts for feature {random_feature_idx} in iteration {iteration+1}:")
        if candidate_counts:
            max_cand = max(candidate_counts)
            counts_array = [candidate_counts.get(i, 0) for i in range(max_cand + 1)]
            print(counts_array)
        else:
            print("No candidates chosen for this feature in this iteration.")
        print(f"Total samples considered for feature {random_feature_idx}: {total_samples_considered}\n")

        
        # --- Update mask: mark samples as fooled if model prediction flips ---
        if model_type == 'keras':
            preds = model.predict(x_test_modified, batch_size=512, verbose=False)
            preds = (preds > 0.5).astype(int).flatten()
        elif model_type == 'torch':
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x_tensor = torch.tensor(x_test_modified, dtype=torch.float32).to(device)
            with torch.no_grad():
                preds = model(x_tensor).cpu().numpy().flatten()
            preds = (preds > 0.5).astype(int)
        mask = (preds != y_test_correct) | mask

        if optimize_already_fooled:
            print("Optimizing samples that were previously fooled")
            x_test_modified = optimize_fooled_samples(
                model, x_test_modified, y_test_correct, mask, min_values, max_values, bin_edges_dict,
                feature_indices=np.arange(x_test_modified.shape[1]), step=step, max_search_steps=5
            )

        # --- Calculate and store fooling ratio ---
        fooling_ratio = np.mean(mask)
        fooling_ratios.append(fooling_ratio)

        # --- Calculate and store JS distances for each feature ---
        js_distances = []
        for feature in bin_edges_dict:
            bin_edges = cp.asarray(bin_edges_dict[feature])
            orig_hist, _ = cp.histogram(cp.asarray(x_test_correct[:, feature]), bins=bin_edges, density=True)
            mod_hist, _ = cp.histogram(cp.asarray(x_test_modified[:, feature]), bins=bin_edges, density=True)

            jsd = compute_js_distance_histogram_gpu(orig_hist, mod_hist, bin_edges)
            js_distances.append(jsd)

        js_distances_per_iteration.append(js_distances)

        # --- Always compute correlation metrics before appending ---
        corr_clean = np.corrcoef(x_test_correct, rowvar=False)
        corr_adv = np.corrcoef(x_test_modified, rowvar=False)
        frob_corr_clean = np.linalg.norm(corr_clean, ord='fro')
        frob_corr_diff = np.linalg.norm(corr_adv - corr_clean, ord='fro')
        rel_corr_change = frob_corr_diff / (frob_corr_clean + 1e-10)
        frob_corr_diffs_per_iteration.append(frob_corr_diff)


        # --- Feature JSD comparison plots (min/median/max) ---
        jsd_per_feature = []
        for feature in bin_edges_dict:
            bin_edges = np.asarray(bin_edges_dict[feature])
            orig_hist, _ = np.histogram(x_test_correct[:, feature], bins=50, density=True)
            adv_hist, _ = np.histogram(x_test_modified[:, feature], bins=50, density=True)
            jsd = jensenshannon(orig_hist, adv_hist, base=2)
            jsd_per_feature.append(jsd)
        jsd_per_feature = np.array(jsd_per_feature)
        min_idx = np.argmin(jsd_per_feature)
        median_idx = np.argsort(jsd_per_feature)[len(jsd_per_feature)//2]
        max_idx = np.argmax(jsd_per_feature)
        feature_indices_plot = [min_idx, median_idx, max_idx]
        feature_titles = [
            f"Feature {min_idx} (min JSD: {jsd_per_feature[min_idx]:.4f})",
            f"Feature {median_idx} (median JSD: {jsd_per_feature[median_idx]:.4f})",
            f"Feature {max_idx} (max JSD: {jsd_per_feature[max_idx]:.4f})"
        ]
        if save_results:
            plt.figure(figsize=(18, 5))
            for idx, feat_idx in enumerate(feature_indices_plot):
                plt.subplot(1, 3, idx+1)
                plt.hist(x_test_correct[:, feat_idx], bins=50, density=True, histtype="step", label="Clean", color='blue')
                plt.hist(x_test_modified[:, feat_idx], bins=50, density=True, histtype="step", label="Adversarial", color='red', linestyle='--')
                plt.title(feature_titles[idx], fontsize=24)
                plt.xlabel("Feature Value", fontsize=24)
                plt.ylabel("Density", fontsize=24)
                plt.tick_params(axis='both', which='major', labelsize=20)
                plt.legend()
            plt.tight_layout()
            plt.savefig(f"{iter_save_dir}/feature_jsd_comparison_iter_{iteration+1}.pdf")
            plt.close()

            # --- Per-class feature JSD comparison plots (min/median/max) ---
            for class_label in [0, 1]:
                class_mask = (y_test_correct == class_label)
                plt.figure(figsize=(18, 5))
                for idx, feat_idx in enumerate(feature_indices_plot):
                    plt.subplot(2, 3, idx+1)
                    plt.hist(x_test_correct[class_mask, feat_idx], bins=50, density=True, histtype="step", label="Clean", color='blue')
                    plt.hist(x_test_modified[class_mask, feat_idx], bins=50, density=True, histtype="step", label="Adversarial", color='red', linestyle='--')
                    plt.title(f"Class {class_label} - {feature_titles[idx]}")
                    plt.xlabel("Feature Value")
                    plt.ylabel("Density")
                    plt.legend()

                    # Calculate bin counts for ratio plot
                    orig_hist, bins = np.histogram(x_test_correct[class_mask, feat_idx], bins=50)
                    adv_hist, _ = np.histogram(x_test_modified[class_mask, feat_idx], bins=bins)
                    # Avoid division by zero
                    ratio = np.divide(adv_hist, orig_hist, out=np.zeros_like(adv_hist, dtype=float), where=orig_hist!=0)
                    uncertainty = np.sqrt(orig_hist + adv_hist) / (orig_hist + adv_hist + 1e-10)  # sqrt(N)/N

                    # Ratio plot below
                    plt.subplot(2, 3, idx+4)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    plt.errorbar(bin_centers, ratio, yerr=uncertainty, fmt='o', color='purple', label='Adv/Clean Ratio')
                    plt.axhline(1.0, color='gray', linestyle='--')
                    plt.xlabel("Feature Value")
                    plt.ylabel("Adv/Clean Ratio")
                    plt.title(f"Class {class_label} - Relative Change")
                    plt.legend()
                plt.tight_layout()
                plt.savefig(f"{iter_save_dir}/feature_jsd_comparison_class{class_label}_iter_{iteration+1}.pdf")
                plt.close()

            # --- Correlation matrix comparison plot ---
            plt.figure(figsize=(14, 6))
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(corr_clean, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(label='Correlation')
            ax1.set_title(f'Clean Correlation Matrix\nIteration {iteration+1}', fontsize=24)
            ax1.tick_params(axis='both', which='major', labelsize=20)
            
            ax2 = plt.subplot(1, 2, 2)
            ax2.imshow(corr_adv, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(label='Correlation')
            ax2.set_title(f'Adversarial Correlation Matrix\nIteration {iteration+1}', fontsize=24)
            ax2.tick_params(axis='both', which='major', labelsize=20)
            
            plt.tight_layout()
            plt.savefig(f"{iter_save_dir}/corr_matrix_comparison_iter_{iteration+1}.pdf")
            plt.close()


        # --- Print summary statistics for this iteration ---
        mean_jsd = np.nanmean(js_distances)
        print(f"Iteration {iteration+1} summary:")
        print(f"  Mean JSD: {mean_jsd:.4f}")
        print(f"  Relative correlation change: {rel_corr_change:.4f}")
        print(f"  Fooling ratio: {fooling_ratio:.4f}")
        print("-" * 40)


        # --- Calculate global constraint metrics ---
        max_jsd_feature = np.max(js_distances)
        corr_clean = np.corrcoef(x_test_correct, rowvar=False)
        corr_adv = np.corrcoef(x_test_modified, rowvar=False)
        frob_corr_clean = np.linalg.norm(corr_clean, ord='fro')
        frob_corr_diff = np.linalg.norm(corr_adv - corr_clean, ord='fro')
        rel_corr_change = frob_corr_diff / (frob_corr_clean + 1e-10)

        
        # Original logic: compare to original distribution
        if track_best_constraint_iter:
            if (max_jsd_feature < constraint_jsd) and (rel_corr_change < constraint_rel_corr):
                if fooling_ratio > best_constraint_fooling:
                    best_constraint_iter = iteration
                    best_constraint_fooling = fooling_ratio
                    best_constraint_jsd = max_jsd_feature
                    best_constraint_rel_corr = rel_corr_change


        iter_end = time.time()
        elapsed = iter_end - iter_start
        samples_per_sec = n_samples / elapsed if elapsed > 0 else float('inf')
        print(f"Iteration {iteration+1}: {n_samples} samples in {elapsed:.2f} s ({samples_per_sec:.2f} samples/sec)")

        
        for var in ['js_distances', 'batch_histograms', 'candidate_changes_padded', 'best_changes']:
            if var in locals():
                del locals()[var]
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()
        gc.collect()


    print(f"Final fooling ratio: {fooling_ratios[-1]}")
    print(f"Final Jensen-Shannon distances per feature: {js_distances_per_iteration[-1]}")
    print(f"Final Frobenius norm of correlation difference: {frob_corr_diffs_per_iteration[-1]}")

    if track_best_constraint_iter:
        return x_test_modified, best_constraint_iter, best_constraint_fooling, best_constraint_jsd, best_constraint_rel_corr
    else:
        return x_test_modified