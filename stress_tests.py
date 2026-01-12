import numpy as np
from scipy.stats import t as student_t

import common
import em
import kmeans

"""
Stress Test 1: Force missing data
"""
def apply_missing(X, missing_rate, seed=0):

    """
    Force missing values at random nonzero locations at X.
    Args:
        X: Original data matrix
        missing_rate: Fraction of nonzero entries to mask as missing
        seed: Random seed for reproducibility
    Returns:
        X_masked: Data matrix with missing entries set to zero

    Keep in mind that missing rate is defined as the fraction of nonzero entries that are masked as missing.
    When compared with original data, the overall fraction of entries that are different will be lower than the missing rate since zero entries are untouched.
    """

    rng = np.random.default_rng(seed)
    X_masked = X.copy()

    observed = X != 0
    idx = np.argwhere(observed)
    n_mask = int(len(idx) * missing_rate)

    mask_idx = rng.choice(len(idx), n_mask, replace=False)
    for i in mask_idx:
        r, c = idx[i]
        X_masked[r, c] = 0

    return X_masked

"""
Stress Test 2: Noise
"""

def add_user_heteroskedastic_noise(X, base_sigma=0.3, alpha=1.0, rng=None):
    if rng is None: rng = np.random.default_rng()
    X_noisy = X.copy()
    observed_mask = X != 0
    
    for u in range(X.shape[0]):
        u_idx = observed_mask[u]
        if u_idx.sum() < 2: continue
        
        # Scale noise by user's internal variance
        sigma_u = base_sigma * (1 + alpha * np.std(X[u, u_idx]))
        noise = rng.normal(0, sigma_u, size=u_idx.sum())
        
        X_noisy[u, u_idx] = np.clip(X[u, u_idx] + noise, 1.0, 5.0)
    return X_noisy

def add_item_correlated_noise(X, rank=5, total_sigma=0.5, rng=None):
    """
    Simulates systematic bias (e.g., all horror movies getting 
    noisier ratings due to a specific sub-genre bias).
    """
    if rng is None: rng = np.random.default_rng()
    n_users, n_items = X.shape
    
    # Generate low-rank noise structure
    U_noise = rng.normal(0, 1, (n_users, rank))
    V_noise = rng.normal(0, 1, (rank, n_items))
    low_rank_noise = (U_noise @ V_noise)
    
    # Standardize and scale the noise
    low_rank_noise /= np.std(low_rank_noise)
    low_rank_noise *= total_sigma
    
    X_noisy = X.copy()
    obs = X != 0
    X_noisy[obs] = np.clip(X[obs] + low_rank_noise[obs], 1.0, 5.0)
    return X_noisy

def apply_mnar_selection_bias(X, selection_strength=2.0, rng=None):
    """
    The probability of a rating being 'missing' depends on its value.
    This breaks the EM assumption that data is MAR (Missing At Random).
    """
    if rng is None: rng = np.random.default_rng()
    X_mnar = X.copy()
    observed_indices = np.argwhere(X != 0)
    
    # Calculate probability of 'staying' observed
    # Users are more likely to report extreme ratings (1 or 5)
    ratings = X[X != 0]
    mean_r = np.mean(ratings)
    # Probability increases as rating deviates from mean
    dist_from_mean = np.abs(ratings - mean_r)
    p_keep = 1 / (1 + np.exp(-selection_strength * dist_from_mean))
    
    # Apply dropout
    keep_mask = rng.random(len(observed_indices)) < p_keep
    to_drop = observed_indices[~keep_mask]
    
    for u, i in to_drop:
        X_mnar[u, i] = 0
        
    return X_mnar

def add_adversarial_flips(X, flip_fraction=0.1, rng=None):
    """
    Specifically targets the EM algorithm by providing 'wrong' labels 
    for the most certain data points.
    """
    if rng is None:
        rng = np.random.default_rng()

    X_flipped = X.copy()
    observed_indices = np.argwhere(X != 0)
    
    if len(observed_indices) == 0:
        return X_flipped

    # Select which indices to corrupt
    n_flips = int(len(observed_indices) * flip_fraction)
    permuted_indices = rng.permutation(len(observed_indices))
    flip_subset = observed_indices[permuted_indices[:n_flips]]

    for u, i in flip_subset:
        old_rating = X[u, i]
        
        # Mapping to create maximum distance from original sentiment
        if old_rating >= 4:
            new_rating = 1.0  # Love -> Hate
        elif old_rating <= 2:
            new_rating = 5.0  # Hate -> Love
        else:
            # For neutral ratings (3), flip to an extreme randomly
            new_rating = rng.choice([1.0, 5.0])
            
        X_flipped[u, i] = new_rating

    return X_flipped

# # Baseline noise
# def add_user_heteroskedastic_noise(X, base_sigma=0.3, alpha=1.2, rng=None):
#     """
#     Add user-dependent Gaussian noise based on observed rating variance.

#     sigma_u = base_sigma * (1 + alpha * std_u)
#     Only observed (non-zero) entries are perturbed.
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     X_noisy = X.copy()
#     observed = X != 0

#     for u in range(X.shape[0]):
#         mask = observed[u]
#         if mask.sum() < 2:
#             continue

#         std_u = np.std(X[u, mask])
#         sigma_u = base_sigma * (1 + alpha * std_u)

#         noise = rng.normal(0, sigma_u, size=mask.sum())
#         X_noisy[u, mask] += noise
#     return X_noisy

# # Moderate noise (item-correlated Gaussian)
# def add_item_correlated_noise(X, rank = 10, base_sigma=0.5, rng=None):
#     """
#     Add item-correlated Gaussian noise to observed (non-zero) entries.
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     n_users, n_items = X.shape
#     X_noisy = X.copy()
#     observed = X != 0

#     Z = rng.normal(0, base_sigma, size = (rank, n_items))

#     for u in range(n_users) :
#         mask = observed[u]
#         if mask.sum() < 2 :
#             continue

#         coeffs = rng.normal(0, 1, size = rank)
#         noise_vector = coeffs @ Z[:, mask]  # shape: (n_items,)

#         X_noisy[u, mask] += noise_vector
#     return X_noisy

# # Strong noise (MNAR removal by rating magnitude) - MORE AGGRESSIVE
# def apply_mnar_missingness_by_rating(X, beta=5.0, drop_fraction=0.3, noise_scale=0.5, rng=None):
#     """
#     Remove observed ratings MNAR based on rating magnitude.
#     Extreme ratings (high |z|) are MORE likely to be kept.
#     Medium ratings near the mean are MORE likely to be dropped.
    
#     Parameters:
#     - beta: steepness of logistic (higher = more extreme selection)
#     - drop_fraction: additional uniform random dropout
#     - noise_scale: scale of Laplace noise added to kept ratings
#     - rng: random number generator
#     Returns:
#     - X_mnar: data matrix with MNAR missingness and noise applied
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     X_mnar = X.copy()
#     observed = X != 0
#     ratings = X[observed]

#     if ratings.size == 0:
#         return X_mnar

#     mu = np.mean(ratings)
#     sigma = np.std(ratings) + 1e-6

#     for u, i in zip(*np.where(observed)):
#         original_rating = X[u, i]

#         # --- MNAR DROPOUT DECISION (CLEAN) ---
#         z = (original_rating - mu) / sigma
#         p_obs = 1 / (1 + np.exp(-beta * abs(z)))

#         if rng.random() > p_obs or rng.random() < drop_fraction:
#             X_mnar[u, i] = 0
#             continue

#         # --- ADD NON-GAUSSIAN NOISE ONLY IF KEPT ---
#         noise = rng.laplace(loc=0.0, scale=noise_scale)
#         X_mnar[u, i] = original_rating + noise

#     return X_mnar

# # Severe noise (heavy-tailed)
# def add_heavy_tailed_noise(X, df=3, scale=3.0, rng=None):
#     if rng is None:
#         rng = np.random.default_rng()

#     X_noisy = X.copy()
#     observed = X != 0

#     if observed.sum() == 0:
#         return X_noisy

#     noise = student_t.rvs(df, size=observed.sum(), random_state=rng) * scale
#     X_noisy[observed] += noise

#     return X_noisy

# # Extreme adversarial noise - MORE AGGRESSIVE
# def add_adversarial_flips(X, flip_fraction=0.2, rng=None):
#     """
#     Adversarially flip 20% of observed ratings.
#     - Ratings of 1-2 → 5 (low to max)
#     - Ratings of 3 → 1 (mid to min, avoid no-op)
#     - Ratings of 4-5 → 1 (high to min)
#     This creates maximum disruption.
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     X_flipped = X.copy()
#     observed = np.argwhere(X != 0)

#     if len(observed) == 0:
#         return X_flipped

#     n_flips = int(len(observed) * flip_fraction)
#     flip_idx = observed[rng.choice(len(observed), n_flips, replace=False)]

#     for u, i in flip_idx:
#         rating = X[u, i]
#         if rating <= 2:
#             X_flipped[u, i] = 5.0  # low → max
#         elif rating == 3:
#             X_flipped[u, i] = 1.0  # mid → min (no no-op)
#         else:  # rating >= 4
#             X_flipped[u, i] = 1.0  # high → min

#     return X_flipped

