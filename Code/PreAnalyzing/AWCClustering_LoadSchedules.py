import numpy as np
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import pandas as pd
from Code.PreAnalyzing.awc import AWC
from itertools import cycle
from sklearn import metrics


def draw(X, labels, name):
    n_clusters = len(set(labels))
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters), colors):
        print(k)
        print(col)
        class_members = labels == k
        plt.plot(X[class_members, 0], X[class_members, 2], col + 'o')

    plt.title(name)
    plt.show()


def load_clustering():
    src = pd.read_csv('data/last_anonym_2017_vartime.csv', sep=';', index_col=0, parse_dates=True)
    outplot_path = ('Output/plot/')
    X = np.array(src)

    lambda_interval = np.linspace(0., 1., 11)
    AWC_object = AWC(speed=1.)
    # To tune parameter \lambda, plot sum of the weights for \lambda 's from some interval
    # and take a value at the end of plateau or before huge jump.
    AWC_object.plot_sum_of_weights(lambda_interval, X)
    l = 0.6
    AWC_object.awc(l, X)
    clusters = AWC_object.get_clusters()
    labels = AWC_object.get_labels()
    draw(X, labels, name='Load Point')

    print('Estimated number of clusters: %d' % len(set(labels)))
    print('cluster sizes: '),
    for c in clusters:
        print(len(c)),
    print("\nV-measure: %0.3f" % metrics.v_measure_score(y, labels))
