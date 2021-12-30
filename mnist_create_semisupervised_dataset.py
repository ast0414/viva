import argparse
import os
import pickle

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def create_semisupervised_datasets(X, y, n_labeled):
    n_x = X.shape[0]
    n_classes = len(np.unique(y))

    # we assume balanced class labels
    assert n_labeled % n_classes == 0, 'n_labeled not divisible by n_classes; cannot assure class balance.'
    n_labeled_per_class = n_labeled // n_classes

    x_labeled = [0] * n_classes
    x_unlabeled = [0] * n_classes
    y_labeled = [0] * n_classes
    y_unlabeled = [0] * n_classes

    for i in range(n_classes):
        idxs = (y == i).nonzero()[0]
        np.random.shuffle(idxs)

        x_labeled[i] = X[idxs][:n_labeled_per_class]
        y_labeled[i] = y[idxs][:n_labeled_per_class]
        x_unlabeled[i] = X[idxs][n_labeled_per_class:]
        y_unlabeled[i] = y[idxs][n_labeled_per_class:]

    # construct new labeled and unlabeled datasets
    X_labeled = np.concatenate(x_labeled, axis=0)
    X_labeled = np.squeeze(X_labeled)
    y_labeled = np.concatenate(y_labeled, axis=0)

    # Its name is 'unlabeled' but it still has targets. Just different name.
    X_unlabeled = np.concatenate(x_unlabeled, axis=0)
    X_unlabeled = np.squeeze(X_unlabeled)
    y_unlabeled = np.concatenate(y_unlabeled, axis=0)

    return X_labeled, X_unlabeled, y_labeled, y_unlabeled


def main(args):

    # set the random seed
    np.random.seed(args.seed)

    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    # rescale the data
    X = X / 255.
    y = y.astype(int)

    # use the traditional train/test split
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # prepare a separate validation set in advance
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=5000, stratify=y_train)

    # create a semi-supervised (partially labeled) train data
    # it is just to separate and re-organize the train set into two subsets
    X_train_labeled, X_train_unlabeled, y_train_labeled, y_train_unlabeled = create_semisupervised_datasets(X_train, y_train, n_labeled=args.n_labeled)

    dataset = {
        "X_train_labeled": X_train_labeled,
        "X_train_unlabeled": X_train_unlabeled,
        "y_train_labeled": y_train_labeled,
        "y_train_unlabeled": y_train_unlabeled,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    os.makedirs(args.save, exist_ok=True)
    with open(os.path.join(args.save, "mnist_{}_labeled.pkl".format(args.n_labeled)), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST Semi-supervised Dataset Preparation')
    parser.add_argument('--n-labeled', type=int, default=1000, help='number of labeled samples (default: 1000)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save', default='./Data', type=str, metavar='PATH', help='save path')

    args = parser.parse_args()
    args.save = os.path.abspath(args.save)

    main(args)
