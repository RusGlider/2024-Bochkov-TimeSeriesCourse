import math

import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """

    ed_dist = 0
    n = len(ts1)
    for i in range(n):
        ed_dist += np.power((ts1[i] - ts2[i]), 2)
    ed_dist = np.sqrt(ed_dist)

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    norm_ed_dist = 0
    n = len(ts1)

    mut1 = np.mean(ts1)
    mut2 = np.mean(ts2)
    sigt1 = np.std(ts1)
    sigt2 = np.std(ts2)
    norm_ed_dist = math.sqrt(abs(2 * n * (1 - ((np.dot(ts1, ts2) - n * mut1 * mut2) / (n * sigt1 * sigt2)))))
    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size

    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    dtw_dist = 0

    n = len(ts1)
    m = len(ts2)
    d = np.zeros((n + 1, m + 1))
    d[:, :] = np.inf
    #d[0, :] = np.inf
    d[0][0] = 0  # np.power((ts1[0] - ts2[0]), 2)
    #r = math.ceil(r * max(n, m) - 1)
    r = int(np.floor(r * len(ts1)))


    for i in range(1, n + 1):
        for j in range(max(1, i - r), min(m, i + r) + 1): #for j in range(1, n + 1):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            d[i][j] = cost + np.min([d[i - 1, j], d[i, j - 1], d[i - 1, j - 1]])
    dtw_dist = d[-1][-1]

    return dtw_dist
