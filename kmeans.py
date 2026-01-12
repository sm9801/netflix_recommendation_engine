"""Mixture model based on kmeans"""
from typing import Tuple

import numpy as np
from scipy.special import logsumexp

from common import GaussianMixture


def log_likelihood(X: np.ndarray, mixture: GaussianMixture) -> float:
    """Compute log-likelihood of data under the Gaussian mixture model.
    
    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture
        
    Returns:
        float: log-likelihood of the data
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    mu, var, pi = mixture
    
    # Compute squared Euclidean distances from each point to each component mean
    # distances: (n, K)
    distances = np.zeros((n, K))
    for k in range(K):
        distances[:, k] = np.sum((X - mu[k])**2, axis=1)
    
    # Log likelihood: sum of log of weighted Gaussian densities
    eps = 1e-8
    safe_var = np.maximum(var, eps)
    # prevent divide by zero
    log_densities = -0.5 * d * np.log(2 * np.pi * safe_var) - distances / (2 * safe_var)
    
    # Weighted by mixture probabilities: log(pi_k) + log p(X|k)
    weighted_log_densities = log_densities + np.log(pi + 1e-16)
    
    # Log-sum-exp for numerical stability
    lse = logsumexp(weighted_log_densities, axis=1)
    
    return np.sum(lse)


def estep(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """E-step: Assigns each datapoint to the gaussian component with the
    closest mean

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples

        """
    n, _ = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))

    for i in range(n):
        tiled_vector = np.tile(X[i, :], (K, 1))
        sse = ((tiled_vector - mixture.mu)**2).sum(axis=1)
        j = np.argmin(sse)
        post[i, j] = 1

    return post


def mstep(X: np.ndarray, post: np.ndarray) -> Tuple[GaussianMixture, float]:
    """M-step: Updates the gaussian mixture. Each cluster
    yields a component mean and variance.

    Args: X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        float: the distortion cost for the current assignment
    """
    n, d = X.shape
    _, K = post.shape

    n_hat = post.sum(axis=0)
    p = n_hat / n

    cost = 0
    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        mu[j, :] = post[:, j] @ X / n_hat[j]
        sse = ((mu[j] - X)**2).sum(axis=1) @ post[:, j]
        cost += sse
        var[j] = sse / (d * n_hat[j])

    return GaussianMixture(mu, var, p), cost


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: distortion cost of the current assignment
    """

    prev_cost = None
    cost = None
    while (prev_cost is None or prev_cost - cost > 1e-4):
        prev_cost = cost
        post = estep(X, mixture)
        mixture, cost = mstep(X, post)

    return mixture, post, cost

def fill_missing_with_kmeans_zeros(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """
    Fill missing values in X using K-Means cluster means, treating 0 as missing.

    Args:
        X: (n, d) data array with missing values represented as 0
        mixture: Trained GaussianMixture object with cluster means

    Returns:
        X_filled: (n, d) data array with missing values filled
    """
    X_filled = X.copy()
    n, d = X.shape
    K, _ = mixture.mu.shape

    for i in range(n):
        row = X[i, :]
        missing_idx = (row == 0)  # treat 0 as missing
        if missing_idx.any():
            observed_idx = ~missing_idx
            if observed_idx.sum() == 0:
                # If entire row is missing, fill with overall mean of all clusters
                X_filled[i, :] = mixture.mu.mean(axis=0)
                continue

            # Compute distance to each cluster using only observed dimensions
            distances = np.sum((mixture.mu[:, observed_idx] - row[observed_idx])**2, axis=1)
            cluster_idx = np.argmin(distances)

            # Fill missing values with cluster mean
            X_filled[i, missing_idx] = mixture.mu[cluster_idx, missing_idx]

    return X_filled
