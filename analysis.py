import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def main():
    plot_codewords_vib()

def plot_codewords_vib():
    cw_dropout = reshape_codewords(np.load("codewords2_with_vib.npy"))
    labels_dropout = np.concatenate(np.load("true_labels_with_vib.npy"))
    cw_no_dropout = reshape_codewords(np.load("codewords2_no_vib.npy"))
    labels_no_dropout = np.concatenate(np.load("true_labels_no_vib.npy"))

    all_labels = np.unique(labels_dropout)
    cw_dropout_class = {x: cw_dropout[labels_dropout == x,:] for x in all_labels}
    cw_no_dropout_class = {x: cw_no_dropout[labels_dropout == x,:] for x in all_labels}

    colors = plt.cm.rainbow(np.linspace(0, 1, all_labels.size))
    for lbl in all_labels:
        sampler = np.random.choice(cw_dropout_class[lbl].shape[0], 100)
        data = cw_dropout_class[lbl][sampler,:]
        plt.scatter(data[:,0], data[:,1], color=colors[lbl], alpha=0.3)
    plt.savefig("plots/codewords_with_vib.png")
    plt.close()

    for lbl in all_labels:
        sampler = np.random.choice(cw_no_dropout_class[lbl].shape[0], 100)
        data = cw_no_dropout_class[lbl][sampler,:]
        plt.scatter(data[:,0], data[:,1], color=colors[lbl], alpha=0.3)
    plt.savefig("plots/codewords_no_vib.png")
    plt.close()


def reshape_codewords(X):
    CW = np.concatenate(X, axis=0)
    return CW.reshape(CW.shape[0], -1)


def compute_within_class_similarity():
    pass


if __name__=="__main__":
    main()
