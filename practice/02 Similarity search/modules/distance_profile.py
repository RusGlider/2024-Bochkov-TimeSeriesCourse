import numpy as np

from modules.utils import z_normalize
from modules.metrics import ED_distance, norm_ED_distance


def brute_force(ts: np.ndarray, query: np.ndarray, is_normalize: bool = True) -> np.ndarray:
    """
    Calculate the distance profile using the brute force algorithm

    Parameters
    ----------
    ts: time series
    query: query, shorter than time series
    is_normalize: normalize or not time series and query

    Returns
    -------
    dist_profile: distance profile between query and time series
    """

    n = len(ts)
    m = len(query)
    N = n-m+1

    dist_profile = np.zeros(shape=(N,))

    _query = query.copy()
    _ts = ts.copy()
    if is_normalize:
        _query = z_normalize(_query)
    for i in range(1, N):
        if is_normalize:
            _ts[i:m+i] = z_normalize(_ts[i:m+i])
        #dist_profile[i] = ED_distance(_query,_ts[i:m+i])
        dist_profile[i] = np.linalg.norm(_query - _ts[i:m+i])

    return dist_profile
