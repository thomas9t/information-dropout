import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def main():
    cw_dropout = reshape_codewords(np.load("temp/codewords2_with_vib.npy"))
    labels_dropout = np.concatenate(np.load("temp/true_labels_with_vib.npy"))
    cw_no_dropout = reshape_codewords(np.load("temp/codewords2_no_vib.npy"))
    labels_no_dropout = np.concatenate(np.load("temp/true_labels_no_vib.npy"))

    all_labels = np.unique(labels_dropout)
    cw_dropout_class = {x: cw_dropout[labels_dropout == x,:] for x in all_labels}
    cw_no_dropout_class = {x: cw_no_dropout[labels_dropout == x,:] for x in all_labels}

    plot_codewords(cw_dropout_class, True)
    plot_codewords(cw_no_dropout_class, False)

    cw_dropout = reshape_codewords(np.load("temp/codewords128_with_vib.npy"))
    labels_dropout = np.concatenate(np.load("temp/true_labels_with_vib.npy"))
    cw_no_dropout = reshape_codewords(np.load("temp/codewords128_no_vib.npy"))
    labels_no_dropout = np.concatenate(np.load("temp/true_labels_no_vib.npy"))

    all_labels = np.unique(labels_dropout)
    cw_dropout_class = {x: cw_dropout[labels_dropout == x,:] for x in all_labels}
    cw_no_dropout_class = {x: cw_no_dropout[labels_dropout == x,:] for x in all_labels}

    similarity_vib = compute_within_class_similarity(cw_dropout_class)
    similarity_no_vib = compute_within_class_similarity(cw_no_dropout_class)


def plot_codewords(cw_classes, vib):
    all_labels = cw_classes.keys()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(all_labels)))
    for lbl in all_labels:
        sampler = np.random.choice(cw_classes[lbl].shape[0], 100)
        data = cw_classes[lbl][sampler,:]
        plt.scatter(data[:,0], data[:,1], color=colors[lbl], alpha=0.3)
    
    vib_stub = "with" if vib else "no"
    plt.savefig("plots/codewords_{}_vib.png".format(vib_stub))
    plt.close()


def reshape_codewords(X):
    CW = np.concatenate(X, axis=0)
    return CW.reshape(CW.shape[0], -1)


def compute_within_class_similarity(cw_classes):
    average_similarities = {lbl: [] for lbl in cw_classes.keys()}
    for lbl in cw_classes.keys():
        for ixa in range(cw_classes[lbl].shape[0]):
            for ixb in range(ixa+1, cw_classes[lbl].shape[0]):
                a = cw_classes[lbl][ixa,:]
                b = cw_classes[lbl][ixb,:]
                eps = a - b
                average_similarities[lbl].append(np.sqrt(np.dot(eps, eps)))
    res = pd.Series(data={x: np.mean(y) for x,y in average_similarities.items()})
    return res

if __name__=="__main__":
    main()
