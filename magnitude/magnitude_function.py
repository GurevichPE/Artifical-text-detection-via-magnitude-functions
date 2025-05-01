import numpy as np
import torch 
from tqdm.auto import tqdm
from collections.abc import Callable

def eucledian_distance(x1:np.ndarray, x2:np.ndarray) -> np.ndarray:
    x1_2 = np.square(x1).sum(axis=1, keepdims=True)
    x2_2 = np.square(x2).sum(axis=1)
    x1_x2 = x1.dot(x2.T)
    return np.sqrt(x1_2 + x2_2 - 2 * x1_x2)


def manhattan_distance(x1:np.ndarray, x2:np.ndarray) -> np.ndarray:
    n_samples, n_features = x1.shape
    return np.abs((x1.reshape(n_samples, 1, n_features,) - x2)).sum(axis=2)


def cosine_distance(x1:np.ndarray, x2:np.ndarray) -> np.ndarray:
    sp = x1.dot(x2.T)
    norm_x1 = np.sqrt(np.square(x1).sum(axis=1))
    norm_x2 = np.sqrt(np.square(x2).sum(axis=1))
    norms = np.matmul(norm_x1.reshape(-1, 1), norm_x2.reshape(1, -1))
    EPS = 1e-8
    cos = sp / (norms + EPS)
    return 1 - cos


def calculate_magnitude(points:np.ndarray|torch.Tensor, distance:Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
    if type(points) == torch.Tensor:
        points = points.numpy()
    dists = distance(points, points)
    Z = np.exp(-dists)
    Z_inv = np.linalg.inv(Z)
    return np.sum(Z_inv)


def calculate_magnitude_function(points:np.ndarray|torch.Tensor, distance:Callable[[np.ndarray, np.ndarray], np.ndarray], T:np.ndarray) -> np.ndarray:
    magnitudes = np.zeros_like(T)
    for i in tqdm(range(len(T))):
        t = T[i]
        t_points = t * points
        mag = calculate_magnitude(t_points, distance)
        magnitudes[i] = mag

    return magnitudes


def calculate_dimension(magnitudes:np.ndarray, t:np.ndarray):
    logt = np.log(t)
    d_logt = logt[1] - logt[0]
    y = np.log(magnitudes)
    grads = np.gradient(y, d_logt)
    return grads


