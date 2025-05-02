import numpy as np
import torch 
from tqdm.auto import tqdm
from collections.abc import Callable
import scipy.spatial.distance as sdist


def calculate_magnitude(points:np.ndarray|torch.Tensor, metric:str) -> float:
    if type(points) == torch.Tensor:
        points = points.numpy()
    dists = sdist.cdist(points, points, metric=metric)
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


