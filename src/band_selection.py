# src/band_selection.py
# Implements band selection methods like SWGMF and E-FDPC.

import numpy as np
import warnings
from sklearn.metrics import pairwise_distances # For E-FDPC
from sklearn.preprocessing import minmax_scale # For E-FDPC normalization
import logging # Use logging

# Ignore potential division by zero warnings during calculations if needed
# warnings.filterwarnings('ignore', category=RuntimeWarning)

# --------------------------------------------------------------------------
#                SWGMF (Sliding Window Grouped Matched Filter)
# --------------------------------------------------------------------------

def _calculate_rmse(band_group):
    """
    Calculates Root Mean Square Error (RMSE) for each band against the group mean band.
    Helper function for SWGMF coarse selection.

    Args:
        band_group (np.ndarray): A subset of bands (H, W, num_bands_in_group).

    Returns:
        tuple: (rmse_per_band, mean_band)
            - rmse_per_band (np.ndarray): RMSE value for each band in the group.
            - mean_band (np.ndarray): The mean band of the group (H, W, 1).
    """
    if band_group.shape[2] == 0:
        return np.array([]), None

    mean_band = np.mean(band_group, axis=2, keepdims=True) # (H, W, 1)
    diff_sq = (band_group - mean_band)**2 # (H, W, num_bands_in_group)
    # Calculate mean squared error across spatial dimensions (H, W) for each band
    mse_per_band = np.mean(diff_sq, axis=(0, 1)) # (num_bands_in_group,)
    rmse_per_band = np.sqrt(mse_per_band)

    return rmse_per_band, mean_band

def _calculate_mf_weights(data_subset):
    """
    Calculates Matched Filter (MF) weights for bands in the subset.
    Helper function for SWGMF fine selection based on paper eq. 2, 3, 4.

    Args:
        data_subset (np.ndarray): Data cube containing the subset of bands (H, W, num_bands).

    Returns:
        np.ndarray: Mean absolute MF weights for each band (num_bands,).
    """
    H, W, num_bands = data_subset.shape
    if num_bands == 0:
        return np.array([])

    # Reshape data for covariance calculation: (num_pixels, num_bands)
    data_reshaped = data_subset.reshape(-1, num_bands) # (H*W, num_bands)

    # Calculate mean vector (m^k in eq 2) - mean of each band across all pixels
    mean_vector = np.mean(data_reshaped, axis=0) # (num_bands,)

    # Calculate covariance matrix (K in eq 2, 3)
    # Add small epsilon for numerical stability during inversion
    cov_matrix = np.cov(data_reshaped.T) + np.eye(num_bands) * 1e-6 # (num_bands, num_bands)

    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix) # K^-1
    except np.linalg.LinAlgError:
        logging.warning("SWGMF: Covariance matrix is singular or near-singular. Using pseudo-inverse.")
        inv_cov_matrix = np.linalg.pinv(cov_matrix) # Use pseudo-inverse if singular

    # Calculate weights for each pixel and each band (w_ijk in eq 2)
    # Center data (x_ijk - m^k)
    centered_data = data_reshaped - mean_vector # (H*W, num_bands)

    # Calculate K^-1 * (x_ijk - m^k) part
    k_inv_x_centered = centered_data @ inv_cov_matrix # (H*W, num_bands)

    # Calculate normalization constant kappa (eq 3)
    # kappa_denom = (x_ijk - m^k)^T * K^-1 * (x_ijk - m^k)
    # Equivalent to element-wise product followed by sum across bands
    kappa_denom = np.sum(centered_data * k_inv_x_centered, axis=1, keepdims=True) # (H*W, 1)

    # Avoid division by zero
    kappa_denom[np.abs(kappa_denom) < 1e-9] = 1e-9 # Avoid exact zero division
    kappa = 1.0 / kappa_denom # (H*W, 1)

    # Full weights: w_ijk = kappa * K^-1 * (x_ijk - m^k)
    weights_all_pixels = kappa * k_inv_x_centered # (H*W, num_bands)

    # Calculate mean absolute weight for each band (|w^k|_mean in eq 4)
    mean_abs_weights = np.mean(np.abs(weights_all_pixels), axis=0) # (num_bands,)

    return mean_abs_weights


def apply_swgmf(data_cube, window_size, target_bands):
    """
    Applies the Sliding Window Grouped Normalized Matched Filter (SWGMF) method.
    Combines coarse selection (RMSE-based grouping) and fine selection (MF ranking).
    Based on paper Section 3.1 and notebook Cell 4.

    Args:
        data_cube (np.ndarray): Input hyperspectral data (H, W, B).
        window_size (int): Size of the sliding window (m) for coarse selection.
        target_bands (int): Desired number of bands after fine selection.

    Returns:
        tuple: (selected_data_cube, selected_indices)
            - selected_data_cube (np.ndarray): Data cube with selected bands (H, W, target_bands).
            - selected_indices (np.ndarray): Indices of the selected bands relative to the original cube.
    """
    H, W, B = data_cube.shape
    representative_band_indices = []
    start_band_index = 0

    logging.info("Starting SWGMF: Coarse Selection (Sliding Window RMSE)...")
    # --- Stage 1: Coarse Dimension Reduction (Sliding Window Grouping) ---
    while start_band_index < B:
        end_band_index = min(start_band_index + window_size, B)
        band_group = data_cube[:, :, start_band_index:end_band_index]

        if band_group.shape[2] == 0:
            break # Should not happen if B > 0

        rmse_values, _ = _calculate_rmse(band_group)

        # Select band with minimum RMSE (most representative of the group mean)
        if len(rmse_values) > 0:
            min_rmse_local_idx = np.argmin(rmse_values)
            # Convert local index within group to global index in original cube
            representative_global_idx = start_band_index + min_rmse_local_idx
            representative_band_indices.append(representative_global_idx)

        # --- Sliding Strategy ---
        # Sliding by 1 ensures overlap and considers all contexts.
        start_band_index += 1

    # Ensure unique and sorted indices from coarse selection
    representative_band_indices = sorted(list(set(representative_band_indices)))
    coarsely_selected_data = data_cube[:, :, representative_band_indices]
    num_coarse_bands = len(representative_band_indices)

    logging.info(f"SWGMF Coarse selection finished. Found {num_coarse_bands} representative bands.")

    # --- Stage 2: Fine Dimension Reduction (Normalized Matched Filter Ranking) ---
    logging.info("Starting SWGMF: Fine Selection (Matched Filter Ranking)...")
    if num_coarse_bands <= target_bands:
        logging.info(f"Number of coarse bands ({num_coarse_bands}) is <= target ({target_bands}). Skipping MF ranking.")
        final_selected_indices = np.array(representative_band_indices)
    else:
        # Calculate MF weights for the coarsely selected bands
        mf_weights = _calculate_mf_weights(coarsely_selected_data) # (num_coarse_bands,)

        # Get indices of bands sorted by descending weight (higher weight = better SNR/quality)
        # Need indices relative to the *coarsely selected* list
        sorted_indices_local = np.argsort(mf_weights)[::-1]

        # Select top 'target_bands' indices from the sorted local indices
        final_selected_indices_local = sorted_indices_local[:target_bands]

        # Map these local indices back to the *original* band indices
        final_selected_indices = np.array(representative_band_indices)[final_selected_indices_local]
        final_selected_indices = np.sort(final_selected_indices) # Sort for consistency

        logging.info(f"SWGMF Fine selection finished. Final selected bands: {len(final_selected_indices)}")

    # Create the final data cube with the selected bands
    selected_data_cube = data_cube[:, :, final_selected_indices]

    # Ensure the number of selected bands matches the target
    if selected_data_cube.shape[-1] != target_bands:
         logging.warning(f"SWGMF returned {selected_data_cube.shape[-1]} bands, but target was {target_bands}. Returning the bands found.")
         # Consider whether to trim/pad or just return what was found. Currently returning what was found.

    return selected_data_cube, final_selected_indices


# --------------------------------------------------------------------------
#                E-FDPC (Enhanced Fast Density Peak Clustering)
# --------------------------------------------------------------------------

def _calculate_cutoff_distance(distances, percent=2.0):
    """
    Calculates the cutoff distance 'dc' based on the pairwise distances.
    FDPC paper suggests choosing dc such that the average number of neighbors
    is around 1-2% of the total number of points.

    Args:
        distances (np.ndarray): Pairwise distance matrix (L x L).
        percent (float): The percentage of neighbors to consider (e.g., 2.0 for 2%).

    Returns:
        float: The calculated cutoff distance dc.
    """
    L = distances.shape[0]
    if L <= 1:
        return 0.0
    # Flatten upper triangle of distance matrix (excluding diagonal)
    upper_triangle_indices = np.triu_indices_from(distances, k=1)
    all_distances = distances[upper_triangle_indices]
    all_distances_sorted = np.sort(all_distances)

    # Position corresponding to the desired percentage
    position = int(len(all_distances_sorted) * (percent / 100.0))
    position = min(position, len(all_distances_sorted) - 1) # Ensure valid index
    position = max(position, 0) # Ensure valid index

    dc = all_distances_sorted[position]
    logging.info(f"E-FDPC Calculated initial cutoff distance dc (at {percent}%): {dc:.4f}")
    return dc

def _assign_clusters(num_bands, nearest_higher_density_neighbor, current_centers_indices):
    """
    Assigns each band to the cluster of its nearest higher-density neighbor
    among the current potential centers. Helper for E-FDPC isolation check.

    Args:
        num_bands (int): Total number of bands (B).
        nearest_higher_density_neighbor (np.ndarray): Array where index i stores the
                                                     index of the nearest band with higher density.
        current_centers_indices (set): Set containing the indices of the bands currently
                                       considered as cluster centers.

    Returns:
        np.ndarray: An array of size num_bands where index i stores the index of the
                    cluster center band it belongs to. -1 if unassigned (should not happen for non-centers).
    """
    cluster_assignment = np.full(num_bands, -1, dtype=int)
    center_set = set(current_centers_indices) # Faster lookups

    # Assign centers to their own cluster
    for center_idx in current_centers_indices:
        cluster_assignment[center_idx] = center_idx

    # Assign non-centers by following the chain
    for i in range(num_bands):
        if i in center_set:
            continue # Skip centers

        current_idx = i
        path = [current_idx]
        while cluster_assignment[current_idx] == -1:
            neighbor_idx = nearest_higher_density_neighbor[current_idx]
            if neighbor_idx == -1: # Reached the highest density point overall
                 # Assign to the highest density center among the current centers if it exists
                 # This case needs careful handling, maybe assign to the globally highest density point if it's a center?
                 # For simplicity, let's find the highest density point among current centers
                 highest_rho_center = -1
                 max_rho = -1
                 # Need rho values here - this approach is getting complex.
                 # Let's rethink the isolation check based on the paper's description.

                 # --- Simpler approach based on paper description ---
                 # "the clustering process of E-FDPC can be stopped if an isolated point has been found."
                 # This implies we check the *newly added* center. If no *other* point's
                 # nearest_higher_density_neighbor points *directly* to this new center, it's isolated.

                 # We don't need full cluster assignment for the isolation check.
                 # We just need to know if any non-center points directly to the candidate center.
                 break # Exit while loop for this point

            if neighbor_idx in center_set:
                # Found a center, assign the whole path to it
                final_center = neighbor_idx
                for node_in_path in path:
                    cluster_assignment[node_in_path] = final_center
                break # Exit while loop for this point
            else:
                # Continue following the chain
                current_idx = neighbor_idx
                if current_idx in path:
                     # Cycle detected - should not happen with FDPC logic if implemented correctly
                     logging.warning(f"E-FDPC: Cycle detected during cluster assignment for band {i}. Path: {path}")
                     break # Avoid infinite loop
                path.append(current_idx)

    return cluster_assignment # This function is actually not needed for the simpler isolation check

def apply_efdpc(data_cube, dc_percent=2.0):
    """
    Applies the Enhanced Fast Density Peak Clustering (E-FDPC) band selection method.
    Automatically determines the number of bands using the isolated-point-stopping criterion.
    Based on the paper by Tang et al. (2015).

    Args:
        data_cube (np.ndarray): Input hyperspectral data (H, W, B).
        dc_percent (float): Percentage used to calculate the initial cutoff distance dc.

    Returns:
        tuple: (selected_data_cube, selected_indices)
            - selected_data_cube (np.ndarray): Data cube with selected bands (H, W, B_selected).
            - selected_indices (np.ndarray): Indices of the selected bands relative to the original cube.
                                             Returns all bands if no isolated point found early.
    """
    logging.info("Starting E-FDPC Band Selection (Automatic Band Count)...")
    H, W, B = data_cube.shape
    if B <= 1:
        logging.warning("E-FDPC: Less than 2 bands available. Skipping selection.")
        return data_cube, np.arange(B)

    # --- Reshape data: Treat each band as a point/vector ---
    # Reshape to (B, H*W) - each row is a flattened band image
    data_reshaped = data_cube.reshape(-1, B).T # Shape (B, H*W)

    # --- Calculate pairwise distances between bands ---
    logging.info("E-FDPC: Calculating pairwise distances between bands...")
    distances = pairwise_distances(data_reshaped, metric='euclidean')
    logging.info(f"E-FDPC: Distance matrix shape: {distances.shape}") # Should be (B, B)

    # --- Calculate initial cutoff distance dc ---
    dc = _calculate_cutoff_distance(distances, dc_percent)

    # --- Calculate rho (local density) and delta (intra-cluster distance) for each band ---
    rho = np.zeros(B)
    delta = np.zeros(B)
    # Stores the index of the nearest band with strictly higher density
    nearest_higher_density_neighbor = np.full(B, -1, dtype=int)

    logging.info("E-FDPC: Calculating rho (local density)...")
    # --- E-FDPC Modification: Adjust dc based on k (number of selected bands) ---
    # We calculate rho iteratively as we select bands (k increases)
    # However, the paper calculates rho once initially. Let's follow the paper's Fig 2a/b logic first,
    # where rho/delta seem calculated once, and gamma is used for ranking.
    # The dc adjustment (Eq 5) might be applied *during* the density calculation if needed,
    # but the paper's figures suggest ranking based on initial rho/delta. Let's stick to that for now.
    dc_new = dc # Use initial dc for density calculation
    for i in range(B):
        # Gaussian kernel for density calculation (E-FDPC Eq. 2)
        rho[i] = np.sum(np.exp(-(distances[i, :] / dc_new)**2)) - 1.0 # Subtract self-distance exp(-(0/dc)^2)=1

    # Sort bands by density in descending order (needed for delta calculation)
    rho_sorted_indices = np.argsort(rho)[::-1]

    logging.info("E-FDPC: Calculating delta (intra-cluster distance)...")
    # Calculate delta (E-FDPC Eq. 3)
    delta.fill(np.max(distances)) # Initialize delta with max distance
    for i in range(1, B): # Iterate through bands sorted by density (excluding the highest density one)
        current_band_idx = rho_sorted_indices[i]
        # Consider only bands with higher density
        higher_density_indices = rho_sorted_indices[:i]

        if len(higher_density_indices) > 0:
            # Find minimum distance to any point with higher density
            dist_to_higher = distances[current_band_idx, higher_density_indices]
            min_dist_idx_local = np.argmin(dist_to_higher)
            delta[current_band_idx] = dist_to_higher[min_dist_idx_local]
            # Store index of nearest higher density neighbor (relative to original indices)
            nearest_higher_density_neighbor[current_band_idx] = higher_density_indices[min_dist_idx_local]
        # else: delta remains max_dist (should only happen for the highest density point)

    # --- Calculate Ranking Score gamma (E-FDPC Eq. 4) ---
    logging.info("E-FDPC: Calculating ranking score gamma...")
    # Normalize rho and delta to [0, 1] before calculating gamma
    rho_norm = minmax_scale(rho)
    # Handle the max distance assigned to the highest density point for delta normalization
    delta_norm = minmax_scale(delta) # Simple min-max scaling should suffice

    # Calculate gamma = rho_norm * (delta_norm^2)
    gamma = rho_norm * (delta_norm ** 2)

    # --- Select Bands using Isolated-Point-Stopping Criterion ---
    logging.info("E-FDPC: Applying isolated-point-stopping criterion...")
    sorted_indices_by_gamma = np.argsort(gamma)[::-1] # Indices sorted by gamma descending

    final_selected_indices = []
    num_selected = 0
    potential_centers = set()

    for k in range(B): # Iterate through bands ordered by gamma
        candidate_idx = sorted_indices_by_gamma[k]
        potential_centers.add(candidate_idx)

        # --- Isolation Check (Simpler Version) ---
        # Check if any *other* band points directly to this candidate as its nearest higher density neighbor
        is_isolated = True
        for other_idx in range(B):
            if other_idx == candidate_idx:
                continue
            # Check if the nearest higher density neighbor of 'other_idx' is the 'candidate_idx'
            if nearest_higher_density_neighbor[other_idx] == candidate_idx:
                is_isolated = False
                break # Found a band pointing to it, not isolated

        if is_isolated and k > 0: # The first point cannot be isolated by this definition
            logging.info(f"E-FDPC: Isolated point found: Band {candidate_idx} (Rank {k+1} by gamma). Stopping selection.")
            num_selected = k # Select bands *before* this isolated one
            final_selected_indices = sorted_indices_by_gamma[:k]
            break # Stop the selection process
        else:
            # If not isolated, or if it's the first band (k=0), keep it for now
            # The actual final list is determined when an isolated point is found
            pass # Continue to the next candidate

        # If loop completes without finding an isolated point (e.g., B=2)
        if k == B - 1:
            logging.warning("E-FDPC: No isolated point found before reaching the end. Selecting all bands.")
            num_selected = B
            final_selected_indices = sorted_indices_by_gamma[:] # Select all

    # --- Finalize Selection ---
    if not isinstance(final_selected_indices, np.ndarray):
         final_selected_indices = np.array(final_selected_indices)

    if len(final_selected_indices) == 0 and B > 0:
         logging.warning("E-FDPC: No bands selected (isolated point might be the first one). Selecting only the top band.")
         final_selected_indices = sorted_indices_by_gamma[:1]
         num_selected = 1

    final_selected_indices = np.sort(final_selected_indices) # Sort indices for consistency

    logging.info(f"E-FDPC finished. Automatically selected {num_selected} bands.")

    # Create the final data cube with the selected bands
    selected_data_cube = data_cube[:, :, final_selected_indices]

    return selected_data_cube, final_selected_indices
