import os
import numpy as np
import random

import pandas as pd
from sktime.distances import euclidean_distance, dtw_distance, pairwise_distance
from sklearn.metrics import silhouette_score
import cv2
import imutils
import glob
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow

from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.pairwise_distance import PairwiseDistance
from modules.clustering import TimeSeriesHierarchicalClustering
from modules.classification import TimeSeriesKNN, calculate_accuracy
#from modules.image_converter import image2ts
from modules.utils import read_ts, z_normalize, sliding_window, random_walk
from modules.plots import plot_ts

def test_distances(dist1: float, dist2: float) -> None:
    """
    Check whether your distance function is implemented correctly

    Parameters
    ----------
    dist1 : distance between two time series calculated by sktime
    dist2 : distance between two time series calculated by your function
    """

    np.testing.assert_equal(round(dist1, 5), round(dist2, 5), 'Distances are not equal')

def task1():
    n = 100
    data1 = random_walk(n)
    data2 = random_walk(n)
    plt.plot(data1)
    plt.plot(data2)
    dist1 = ED_distance(data1, data2)
    dist2 = euclidean_distance(data1, data2)

    test_distances(dist1, dist2)
    print(f'{dist1=}\n{dist2=}')

def task2():
    n = 100
    data1 = random_walk(n)
    data2 = random_walk(n)
    plt.plot(data1)
    plt.plot(data2)
    dist1 = DTW_distance(data1, data2)
    dist2 = dtw_distance(data1, data2)

    test_distances(dist1, dist2)
    print(f'{dist1=}\n{dist2=}')

def test_matrices(matrix1 : np.ndarray, matrix2 : np.ndarray) -> None:
    """
    Check whether your matrix function is implemented correctly

    Parameters
    ----------
    matrix1 : distance matrix calculated by sktime
    matrix2 : distance matrix calculated by your function
    """

    np.testing.assert_equal(matrix1.round(5), matrix2.round(5), 'Matrices are not equal')

def task3():
    n = 100
    pwd = PairwiseDistance(metric='euclidean')

    tss = np.array([random_walk(n) for i in range(10)])

    ed_matrix = pwd.calculate(tss)
    ed_matrix2 = pairwise_distance(tss)

    test_matrices(ed_matrix, ed_matrix2)

def task4(): #TODO
    url = './datasets/part1/CBF_TRAIN.txt'

    data = read_ts(url)
    data = pd.DataFrame(data)
    ts_set = data.iloc[:, 1:]
    labels = data.iloc[:, 0]

    #plot_ts(np.asarray(ts_set))


    dist_matrix_dtw = PairwiseDistance(metric='dtw').calculate(ts_set.values)
    clustering_dtw = TimeSeriesHierarchicalClustering(n_clusters=3, method='average')
    dtw_classes = clustering_dtw.fit_predict(dist_matrix_dtw)
    #clustering_dtw.plot_dendrogram(ts_set.values, dtw_classes, title='DTW дендрограмма')

    dist_matrix_eucl = PairwiseDistance(metric='euclidean').calculate(ts_set.values)
    clustering_eucl = TimeSeriesHierarchicalClustering(n_clusters=3, method='average')
    eucl_classes = clustering_eucl.fit_predict(dist_matrix_eucl)
    #clustering_eucl.plot_dendrogram(ts_set.values, eucl_classes, title='Euclidean дендрограмма')

    silhouette_dtw = silhouette_score(ts_set.values, dtw_classes)
    silhouette_eucl = silhouette_score(ts_set.values, eucl_classes)
    print(f'dtw silhouette score:\t{silhouette_dtw}')
    print(f'eucl silhouette score:\t{silhouette_eucl}')

    #dist_matrix = PairwiseDistance(metric='euclidean').calculate(np.asarray(ts_set))
    #model = TimeSeriesHierarchicalClustering(n_clusters=3).fit(dist_matrix)
    #model.plot_dendrogram(ts_set,labels)
    #clust_fit = clust.fit(PairwiseDistance(metric="dtw").calculate(np.asarray(ts_set)))
    #clust_fit.plot_dendrogram(np.asarray(ts_set), labels)


def task5():
    n = 100
    data1 = random_walk(n)
    data2 = random_walk(n)

    dist1 = norm_ED_distance(data1,data2)
    dist2 = euclidean_distance(z_normalize(data1),z_normalize(data2))
    test_distances(dist1,dist2)
    print('test complete')

def task6():
    url1 = './datasets/part2/chf10.csv'
    ts1 = read_ts(url1)

    url2 = './datasets/part2/chf11.csv'
    ts2 = read_ts(url2)

    ts_set = np.concatenate((ts1, ts2), axis=1).T

    plot_ts(ts_set)

    m = 125
    subs_set1 = sliding_window(ts_set[0], m, m - 1)
    subs_set2 = sliding_window(ts_set[1], m, m - 1)

    subs_set = np.concatenate((subs_set1[0:15], subs_set2[0:15]))
    labels = np.array([0] * subs_set1[0:15].shape[0] + [1] * subs_set2[0:15].shape[0])


    pairwise_euclidean = PairwiseDistance(metric='euclidean', is_normalize=False)
    distance_matrix_euclidean = pairwise_euclidean.calculate(subs_set)
    clustering_euclidean = TimeSeriesHierarchicalClustering(n_clusters=2, method='complete').fit(distance_matrix_euclidean)


    pairwise_norm_euclidean = PairwiseDistance(metric='euclidean', is_normalize=True)
    distance_matrix_norm_euclidean = pairwise_norm_euclidean.calculate(subs_set)
    clustering_norm_euclidean = TimeSeriesHierarchicalClustering(n_clusters=2, method='complete').fit(distance_matrix_norm_euclidean)


    # Визуализируем результаты для обычной евклидовой метрики
    clustering_euclidean.plot_dendrogram(subs_set, labels, title='Dendrogram for Euclidean Distance')

    # Визуализируем результаты для нормализованной евклидовой метрики
    clustering_norm_euclidean.plot_dendrogram(subs_set, labels, title='Dendrogram for Normalized Euclidean Distance')

    eucl_prediction = clustering_euclidean.model.labels_ #clustering_euclidean.fit_predict(distance_matrix_euclidean)
    eucl_norm_prediction = clustering_norm_euclidean.model.labels_ #clustering_norm_euclidean.fit_predict(distance_matrix_norm_euclidean)

    silhouette_norm_eucl = silhouette_score(subs_set,eucl_norm_prediction)
    silhouette_eucl = silhouette_score(subs_set,eucl_prediction)

    print(f'eucl silhouette score:\t\t{silhouette_eucl}')
    print(f'eucl norm silhouette score:\t\t{silhouette_norm_eucl}')

def task7():
    pass

def main():
    n = 100
    data1 = random_walk(n)
    data2 = random_walk(n)
    plt.plot(data1)
    plt.plot(data2)

    dist1 = DTW_distance(data1,data2)
    dist2 = dtw_distance(data1,data2)
    print(f'{dist1=}\n{dist2=}')
    test_distances(dist1, dist2)


if __name__ == '__main__':
    task6()