"""Mixture model for matrix completion"""
from typing import Tuple

import numpy as np
from scipy.special import logsumexp

from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    # class GaussianMixture(NamedTuple):
    #     """Tuple holding a gaussian mixture"""
    #     mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    #     var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    #     p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component

    K = mixture.mu.shape[0]
    n, d = X.shape
    mu, var, pi = mixture

    identity_matrix = X.astype(bool).astype(int)

    f = (np.sum(X ** 2, axis = 1)[:, None] + np.matmul(identity_matrix, mu.T ** 2) - 2 * np.matmul(X, mu.T)) / (2 * var)
    coefficient = np.matmul((-np.sum(identity_matrix, axis = 1).reshape(-1, 1) / 2.0), (np.log((2 * np.pi * var)).reshape(-1, 1)).T)
    f = coefficient - f

    f = f + np.log(pi + 1e-16)

    lse = logsumexp(f, axis = 1).reshape(-1, 1)
    log_posterior = f - lse

    LL = np.sum(lse, axis = 0).item()

    return [np.exp(log_posterior), LL]
    raise NotImplementedError



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25, max_variance: float = 3.0,
          robust: bool = True) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian
        max_variance: the maximum variance allowed (robustness guard)
        robust: if True, downweight outliers using Tukey bisquare weights

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    mu_hat = mixture.mu.copy()
    n_hat = np.sum(post, axis = 0)
    p_hat = n_hat / n

    identity_matrix = X.astype(bool).astype(int)

    mu_numerator = np.matmul(np.transpose(post), X)
    mu_denominator = np.matmul(np.transpose(post), identity_matrix)

    target = np.where(mu_denominator >= 1)

    mu_hat[target] = mu_numerator[target] / mu_denominator[target]

    norm_xmu = np.sum(X ** 2, axis = 1)[:, None] + np.matmul(identity_matrix, np.transpose(mu_hat) ** 2) - 2 * np.matmul(X, np.transpose(mu_hat))

    if robust:
        # Tukey bisquare weights to downweight large residuals; vectorized over components
        residuals = np.sqrt(np.maximum(norm_xmu, 0))  # (n, K)
        med = np.median(residuals, axis=0)
        mad = np.median(np.abs(residuals - med), axis=0) + 1e-10  # avoid divide-by-zero
        c = 4.685
        u = np.abs(residuals - med) / (c * mad)
        weights = np.where(u <= 1.0, (1.0 - u ** 2) ** 2, 0.0)
        norm_xmu *= weights

    sigma_denominator = np.sum(np.sum(identity_matrix, axis = 1).reshape(-1, 1) * post, axis = 0)
    sigma_hat = np.maximum(np.sum(post * norm_xmu, axis = 0) / sigma_denominator, min_variance)
    sigma_hat = np.minimum(sigma_hat, max_variance)

    return GaussianMixture(mu_hat, sigma_hat, p_hat)
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray, robust: bool = True,
        max_variance: float = 3.0) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples
        robust: if True, enable robust weighting in the M-step
        max_variance: upper cap on variance per component

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    log_old = 0
    log_new = 0

    while log_old == 0 or ((log_new - log_old) > (10 ** (-6)) * np.abs(log_new)):
        log_old = log_new

        post = estep(X, mixture)[0]
        log_new = estep(X, mixture)[1]

        mixture = mstep(X, post, mixture, robust=robust, max_variance=max_variance)

    return mixture, post, log_new
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    K = mixture.mu.shape[0]
    n, d = X.shape
    mu, var, pi = mixture

    identity_matrix = X.astype(bool).astype(int)

    f = (np.sum(X ** 2, axis=1)[:, None] + np.matmul(identity_matrix, mu.T ** 2) - 2 * np.matmul(X, mu.T)) / (2 * var)
    coefficient = np.matmul((-np.sum(identity_matrix, axis=1).reshape(-1, 1) / 2.0),
                            (np.log((2 * np.pi * var)).reshape(-1, 1)).T)
    f = coefficient - f

    f = f + np.log(pi + 1e-16)

    lse = logsumexp(f, axis=1).reshape(-1, 1)
    log_posterior = f - lse


    X_new = X.copy()
    mu, sigma, p = mixture
    target = np.where(X == 0)
    X_new[target] = np.matmul(np.exp(log_posterior), mu)[target]
    return X_new
    raise NotImplementedError

