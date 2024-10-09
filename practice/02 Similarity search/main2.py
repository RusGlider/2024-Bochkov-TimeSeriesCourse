import os

import numpy as np
import pandas as pd
import math
import timeit
import random
import mass_ts as mts
#from IPython.display import display

from modules.distance_profile import brute_force
from modules.prediction import *
from modules.bestmatch import *
from modules.utils import *
from modules.plots import *


def task1():
    ts_url = './datasets/part1/ECG.csv'
    query_url = './datasets/part1/ECG_query.csv'

    ts = read_ts(ts_url).reshape(-1)
    query = read_ts(query_url).reshape(-1)

    plot_bestmatch_data(ts, query)


    topK = 2
    excl_zone_frac = 0.5
    excl_zone = math.ceil(len(query) * excl_zone_frac)
    is_normalize = True

    naive_bestmatch_results = {}


    distance_profile = brute_force(ts,query,is_normalize)

    #distances = mts.mass(ts, query)
    #print(f'{distance_profile=}')
    #print(f'{distances=}')

    naive_bestmatch_results = topK_match(distance_profile,excl_zone,topK)
    #for k, v in naive_bestmatch_results.items():
        #print(k,v)

    print(naive_bestmatch_results)
    plot_bestmatch_results(ts, query, naive_bestmatch_results)

def task2():
    import warnings
    np.warnings = warnings

    ts_url = './datasets/part1/ECG.csv'
    query_url = './datasets/part1/ECG_query.csv'

    ts = read_ts(ts_url).reshape(-1)
    query = read_ts(query_url).reshape(-1)

    topK = 2
    excl_zone_frac = 0.5
    excl_zone = math.ceil(len(query) * excl_zone_frac)
    is_normalize = True

    #distances = mts.mass(ts, query)
    #top_motifs = mts.top_k_motifs(distances, topK, excl_zone)
    #print(f'{distances=}')
    #print(f'{top_motifs=}')

    indices, distances = mts.mass2_batch(z_normalize(ts), z_normalize(query), 500, top_matches=topK, n_jobs=1)
    mass_dict = {'indices': indices.tolist()}
    plot_bestmatch_results(ts, query, mass_dict)

if __name__ == '__main__':
    task2()