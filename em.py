"""Mixture model for matrix completion"""
from typing import Tuple

import numpy as np
from scipy.special import logsumexp

from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component
    
    SAFEGUARDS:
    - Added 1e-16 to variance denominators to prevent division by zero.
    - Added 1e-16 to log calculations.
    """
    K = mixture.mu.shape[0]
    n, d = X.shape
    mu, var, pi = mixture

    # Prevent variance from being zero (numerical stability)
    var = np.maximum(var, 1e-16)

    identity_matrix = X.astype(bool).astype(int)

    # SAFEGUARD: Add epsilon to denominator
    f = (np.sum(X ** 2, axis=1)[:, None] + 
         np.matmul(identity_matrix, mu.T ** 2) - 
         2 * np.matmul(X, mu.T)) / (2 * var + 1e-16)
         
    # SAFEGUARD: Add epsilon to log
    coefficient = np.matmul((-np.sum(identity_matrix, axis=1).reshape(-1, 1) / 2.0), 
                            (np.log((2 * np.pi * var) + 1e-16).reshape(-1, 1)).T)
    
    f = coefficient - f
    f = f + np.log(pi + 1e-16)  # pi is already safeguarded in your original code, kept it.

    lse = logsumexp(f, axis=1).reshape(-1, 1)
    log_posterior = f - lse
    
    LL = np.sum(lse, axis=0).item()

    return np.exp(log_posterior), LL


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25, max_variance: float = 3.0,
          robust: bool = True) -> GaussianMixture:
    """M-step: Updates the gaussian mixture
    
    SAFEGUARDS:
    - Maintained masking logic for mu updates (prevents div by 0 for empty clusters).
    - Added epsilon to MAD (Median Absolute Deviation) to prevent robust weight crash.
    - Added epsilon to sigma_denominator.
    """
    n, d = X.shape
    mu_hat = mixture.mu.copy()
    n_hat = np.sum(post, axis=0)
    p_hat = n_hat / n

    identity_matrix = X.astype(bool).astype(int)

    mu_numerator = np.matmul(np.transpose(post), X)
    mu_denominator = np.matmul(np.transpose(post), identity_matrix)

    # Logic check: Only update means where we have observed data
    target = np.where(mu_denominator >= 1)
    mu_hat[target] = mu_numerator[target] / mu_denominator[target]

    norm_xmu = np.sum(X ** 2, axis=1)[:, None] + \
               np.matmul(identity_matrix, np.transpose(mu_hat) ** 2) - \
               2 * np.matmul(X, np.transpose(mu_hat))

    if robust:
        # SAFEGUARD: maximize(..., 0) ensures no negative values inside sqrt due to float errors
        residuals = np.sqrt(np.maximum(norm_xmu, 0)) 
        med = np.median(residuals, axis=0)
        
        # SAFEGUARD: Add epsilon to MAD to avoid division by zero if all residuals are identical
        mad = np.median(np.abs(residuals - med), axis=0) + 1e-10
        
        c = 4.685
        u = np.abs(residuals - med) / (c * mad)
        weights = np.where(u <= 1.0, (1.0 - u ** 2) ** 2, 0.0)
        norm_xmu *= weights

    # SAFEGUARD: Add epsilon to denominator
    sigma_denominator = np.sum(np.sum(identity_matrix, axis=1).reshape(-1, 1) * post, axis=0)
    sigma_hat = np.maximum(np.sum(post * norm_xmu, axis=0) / (sigma_denominator + 1e-8), min_variance)
    
    # Enforce variance caps
    sigma_hat = np.minimum(sigma_hat, max_variance)

    return GaussianMixture(mu_hat, sigma_hat, p_hat)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray, robust: bool = True,
        max_variance: float = 3.0) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model
    
    OPTIMIZATION:
    - Fixed inefficiency where estep() was called twice per loop.
    """
    log_old = None
    log_new = None
    
    # Initialize with first E-step
    post, log_new = estep(X, mixture)

    # Use a loop limit or epsilon check
    while log_old is None or ((log_new - log_old) > 1e-6 * np.abs(log_new)):
        log_old = log_new
        
        mixture = mstep(X, post, mixture, robust=robust, max_variance=max_variance)
        post, log_new = estep(X, mixture)

    return mixture, post, log_new


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model
    
    SAFEGUARDS:
    - Applied same var/log protections as estep.
    """
    mu, var, pi = mixture
    
    # Prevent variance from being zero
    var = np.maximum(var, 1e-16)

    identity_matrix = X.astype(bool).astype(int)

    # SAFEGUARD: Add epsilon
    f = (np.sum(X ** 2, axis=1)[:, None] + 
         np.matmul(identity_matrix, mu.T ** 2) - 
         2 * np.matmul(X, mu.T)) / (2 * var + 1e-16)
         
    # SAFEGUARD: Add epsilon
    coefficient = np.matmul((-np.sum(identity_matrix, axis=1).reshape(-1, 1) / 2.0),
                            (np.log((2 * np.pi * var) + 1e-16).reshape(-1, 1)).T)
                            
    f = coefficient - f
    f = f + np.log(pi + 1e-16)

    lse = logsumexp(f, axis=1).reshape(-1, 1)
    log_posterior = f - lse

    X_new = X.copy()
    
    # Only fill missing entries (where X == 0)
    target = np.where(X == 0)
    
    # Calculate weighted average of means based on posterior probabilities
    predicted_means = np.matmul(np.exp(log_posterior), mu)
    
    X_new[target] = predicted_means[target]
    
    return X_new