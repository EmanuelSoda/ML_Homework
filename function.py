from sklearn.metrics import silhouette_score
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold


# functions
def visualizeCorrMat(data):  # Create correlation matrixc
    f, ax = plt.subplots(figsize=(20, 20))
    mask = np.triu(np.ones_like(data, dtype=np.bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(data, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.9, cbar_kws={"shrink": .5})


def visualizeCorTarget(target, threshold, data, item):
    relevant_feature = target[target == threshold]
    relevant_feature.index
    max(target)
    sns.pairplot(data, x_vars=relevant_feature.index,
                 y_vars=item, size=5, aspect=1.5)  # , kind = 'reg'
    plt.tight_layout()


def varianceVisualization(data):
    print("Max Variance value :\t", data.var().max())
    print("Min Variance value :\t", data.var().min())
    print("Mean Variance value:\t", data.var().mean())
    plt.subplots(figsize=(20, 4))
    plot = sns.lineplot(x=data.var().index, y=data.var().values)
    plot.set_xticklabels(data, rotation=90)


# compute PC and % variation expressed by
def principalComponent(data, n_comp):
    #scaled_Train = scale(train.drop(['class'], axis = 1))
    pca = PCA(n_comp)
    return pca.fit_transform(data), np.round(100 * pca.explained_variance_ratio_, decimals=2), pca.singular_values_


def visualizeComponentVariance(data, ylab, title, cum=False, per_var=0):
    plot_labels = ['PC' + str(s) for s in range(1, len(data) + 1)]
    plt.figure(figsize=(44, 10))
    if(cum == True):
        colormat = np.where(data > 90, "#9b59b6", "#3498db")
    else:
        colormat = np.where(per_var < 1, "#9b59b6", "#3498db")
    plt.bar(x=range(1, len(data) + 1), height=data,
            tick_label=plot_labels, color=colormat)
    #plt.hlines(y = 90, xmin = 1, xmax = len(cum_var) + 1)
    plt.ylabel(ylab, size=30)
    plt.xlabel('Principal Component', size=30)
    plt.xticks(rotation=45, size=25)
    plt.yticks(size=25)
    plt.title(title, size=40)
    plt.show()
    return plot_labels


def plotPCA(Label, pc, **label_):
    unique_labelClass = np.unique(Label)
    nlabels = len(unique_labelClass)
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.cm.get_cmap('Set2', nlabels)
    smap = cm.ScalarMappable(norm=mcolors.Normalize(unique_labelClass.min(),
                                                    unique_labelClass.max() + 1),   cmap=cmap)
    ax.scatter(xs=pc.PC1, ys=pc.PC2, zs=pc.PC3, marker='o', s=25,
               c=Label, cmap=cmap, )
    ax.set_xlabel(pc.PC1.name, size=15)
    ax.set_ylabel(pc.PC2.name, size=15)
    ax.set_zlabel(pc.PC3.name, size=15)
    cbar = plt.colorbar(mappable=smap, label='Label')


def plot2dPCA(Label, pc1, pc2):
    unique_labelClass = np.unique(Label)
    nlabels = len(unique_labelClass)
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    cmap = plt.cm.get_cmap('Set2', nlabels)
    smap = cm.ScalarMappable(norm=mcolors.Normalize(unique_labelClass.min(),
                                                    unique_labelClass.max() + 1), cmap=cmap)
    plt.scatter(x=pc1, y=pc2, marker='o', s=25, c=Label, cmap=cmap)
    ax.set_xlabel(pc1.name, size=15)
    ax.set_ylabel(pc2.name, size=15)
    plt.colorbar(mappable=smap, label='Label Class')


# metrics

def sorted_sim(sim, y_pred):
    idx_sorted = np.argsort(y_pred)
    # Sort the rows
    sim = sim[idx_sorted]
    # Sort the columns
    sim = sim[:, idx_sorted]

    return sim


def wss(X, y_pred, metric):
    ncluster = np.unique(y_pred).shape[0]
    err = 0
    for k in range(ncluster):
        # All the points of this cluster
        X_k = X[y_pred == k]
        # Distances of all points within the cluster
        dist_mat = pairwise_distances(X_k, metric=metric)
        # Select the lower triangular part of the matrix
        triu_idx = np.tril_indices(dist_mat.shape[0], k=1)
        err += (dist_mat[triu_idx] ** 2).sum()

    return err


def bss(X, y_pred, metric):
    ncluster = np.unique(y_pred).shape[0]
    # Sort the distance matrix (as we did for the simiarity)
    dist_mat = pairwise_distances(X, metric=metric) ** 2
    dist_mat = sorted_sim(dist_mat, y_pred)
    y_sort = np.sort(y_pred)

    err = 0
    for k in range(ncluster):
        kidx = np.where(y_sort == k)[0]
        start, end = kidx[0], kidx[-1]
        err += dist_mat[start:end, end + 1:].sum()

    return err


def incidence_mat(y_pred):
    npoints = y_pred.shape[0]
    mat = np.zeros([npoints, npoints])
    # Retrieve how many different cluster ids there are
    clusters = np.unique(y_pred)
    nclusters = clusters.shape[0]

    for i in range(nclusters):
        sample_idx = np.where(y_pred == i)
        # Compute combinations of these indices
        idx = np.meshgrid(sample_idx, sample_idx)
        mat[idx[0].reshape(-1), idx[1].reshape(-1)] = 1

    return mat


def similarity_mat(X, metric):
    dist_mat = pairwise_distances(X, metric=metric)
    min_dist, max_dist = dist_mat.min(), dist_mat.max()

    sim_mat = 1 - (dist_mat - min_dist) / (max_dist - min_dist)
    return sim_mat


def correlation(X, y_pred, metric):
    inc = incidence_mat(y_pred)
    sim = similarity_mat(X, metric)
    inc = normalize(inc.reshape(1, -1))
    sim = normalize(sim.reshape(1, -1))
    corr = (inc @ sim.T)
    return corr[0, 0]


def sorted_sim(sim, y_pred):
    idx_sorted = np.argsort(y_pred)
    # Sort the rows
    sim = sim[idx_sorted]
    # Sort the columns
    sim = sim[:, idx_sorted]

    return sim


def plot_sorted_sim(sim, y_pred):
    sim = sorted_sim(sim, y_pred)

    fig, ax = plt.subplots(figsize=(40, 30))
    ax = sns.heatmap(sim, ax=ax)
    # Remove ruler (ticks)
    ax.set_yticks([])
    ax.set_xticks([])


# to plot the Metrics


def plotMetrics(X, models):
    silhouette_list, wss_list, bss_list = [], [], []
    for model in models:
        wss_list.append(wss(X, model.fit_predict(X), 'euclidean'))
        bss_list.append(bss(X, model.fit_predict(X), 'euclidean'))
        if model.n_clusters > 1:
            silhouette_list.append(silhouette_score(
                X, model.fit_predict(X), metric='euclidean'))

    plt.plot(list(range(1, len(models) + 1)), wss_list, label='WSS', color='g')
    plt.plot(list(range(1, len(models) + 1)), wss_list, marker='o', color='g')
    plt.plot(list(range(1, len(models) + 1)), bss_list, label='BSS', color='r')
    plt.plot(list(range(1, len(models) + 1)), bss_list, marker='o', color='r')
    plt.legend()
    plt.show()
    plt.plot(list(range(2, len(models) + 1)), silhouette_list,
             label='Silhuette score', color='b')
    plt.plot(list(range(2, len(models) + 1)),
             silhouette_list, marker='o', color='b')
    plt.legend()
    plt.show()


def selectFeatures(X, y):
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(10),
                  scoring='accuracy')
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure(figsize=(15, 10))
    plt.xlabel("Number of features selected", size=15)
    plt.ylabel("Cross validation score (# of correct classifications)",  size=15)
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.plot(range(1, len(rfecv.grid_scores_) + 1),
             rfecv.grid_scores_, marker='o')
    plt.show()
    return rfecv
