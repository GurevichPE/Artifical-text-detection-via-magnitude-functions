# magnitude_function.py
import numpy as np
import torch
import scipy.spatial.distance as sdist
from typing import Union
from scipy.spatial.distance import cdist

def calculate_magnitude(points: Union[np.ndarray, torch.Tensor], metric: str) -> float:
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    dists = sdist.cdist(points, points, metric=metric)
    Z = np.exp(-dists)
    return float(np.sum(np.linalg.inv(Z)))

def calculate_magnitude_function(
    points: Union[np.ndarray, torch.Tensor],
    metric: str,
    T: np.ndarray
) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    mags = np.zeros(len(T), dtype=float)
    for i, t in enumerate(T):
        mags[i] = calculate_magnitude(t * points, metric)
    return mags

def calculate_dimension(magnitudes: np.ndarray, t: np.ndarray) -> np.ndarray:
    logt = np.log(t)
    return np.gradient(np.log(magnitudes), logt[1] - logt[0])

#New way to compute z inverse
def gradient_z_inverse(points: np.ndarray, metric: str = "cityblock",
                                  tol: float = 1e-6, max_iter: int = 1000) -> float:
    D = cdist(points, points, metric=metric)
    Z = np.exp(-D)

    n = Z.shape[0]
    b = np.ones(n)
    w = np.zeros(n)
    r = b - Z @ w
    p = r.copy()
    rho = np.dot(r, r)

    for k in range(max_iter):
        q = Z @ p
        alpha = rho / np.dot(p, q)
        w = w + alpha * p
        r = r - alpha * q
        rho_new = np.dot(r, r)
        if np.sqrt(rho_new) < tol * np.sqrt(n):
            break
        beta = rho_new / rho
        p = r + beta * p
        rho = rho_new

    return float(np.sum(w))
