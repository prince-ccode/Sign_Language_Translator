#!/usr/bin/env python3
"""
evaluate.py
Small evaluation helper for the sign language project.

Usage examples (from activated venv):
  # Evaluate a saved model on a held-out portion of data
  python evaluate.py --data mydata.npz --model model.pkl --split 0.2

  # If you already trained a model using the project's train command and want
  # to evaluate it on all samples (no split) use split=0 or omit --split.

  # Perform k-fold cross validation on the dataset (no saved model required):
  python evaluate.py --data mydata.npz --kfold 5

The script prints overall accuracy, per-class precision/recall, and a confusion matrix.
"""

import argparse
import numpy as np
import pickle
from collections import Counter
import sys
import os

# Import the classifier class from the project (reuses existing implementation)
try:
    from SIgn_LangV1 import KNNClassifier
except Exception:
    # If import fails (e.g., different module name), fall back to a minimal KNN
    KNNClassifier = None


def confusion_matrix(y_true, y_pred, labels):
    idx = {lab: i for i, lab in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[idx[t], idx[p]] += 1
    return mat


def precision_recall_from_cm(cm):
    # cm: rows true, cols pred
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    actual = cm.sum(axis=1).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, 0.0)
    return precision, recall


def evaluate_model_on_split(clf, X_train, y_train, X_test, y_test):
    # If clf is None, train a new KNNClassifier using default k=3
    if clf is None:
        if KNNClassifier is None:
            raise RuntimeError('KNNClassifier not available for training')
        clf = KNNClassifier(k=3)
        clf.fit(X_train, y_train)
    # if model was loaded it must already be fitted
    y_pred = clf.predict(X_test)
    labels = sorted(list(set(list(y_test) + list(y_pred))))
    cm = confusion_matrix(y_test, y_pred, labels)
    precision, recall = precision_recall_from_cm(cm)
    acc = float((np.array(y_pred) == np.array(y_test)).sum()) / len(y_test)
    return acc, labels, cm, precision, recall


def kfold_cv(X, y, k=5, classifier_ctor=None, seed=0):
    n = len(y)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    accs = []
    cms = None
    labels_all = sorted(list(set(y)))
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.hstack([f for j, f in enumerate(folds) if j != i])
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        if classifier_ctor is None:
            if KNNClassifier is None:
                raise RuntimeError('No classifier available for training')
            clf = KNNClassifier(k=3)
        else:
            clf = classifier_ctor()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = float((np.array(y_pred) == np.array(y_test)).sum()) / len(y_test)
        accs.append(acc)
        cm = confusion_matrix(y_test, y_pred, labels_all)
        cms = cm if cms is None else (cms + cm)
    mean_acc = float(np.mean(accs))
    return mean_acc, labels_all, cms


def main():
    p = argparse.ArgumentParser(description='Evaluate saved model or dataset for accuracy.')
    p.add_argument('--data', default='data.npz', help='Path to dataset (.npz)')
    p.add_argument('--model', default=None, help='Path to saved model (model.pkl)')
    p.add_argument('--split', type=float, default=0.2, help='Fraction to hold out for testing (0..1). If 0, use all samples.')
    p.add_argument('--kfold', type=int, default=0, help='If >1, run k-fold cross-validation instead of using --model')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    if not os.path.exists(args.data):
        print('Data file not found:', args.data)
        sys.exit(1)

    d = np.load(args.data, allow_pickle=True)
    X = d['X']
    y = d['y']

    if args.kfold and args.kfold > 1:
        print(f'Running {args.kfold}-fold cross validation on {len(y)} samples...')
        mean_acc, labels, cm = kfold_cv(X, y, k=args.kfold, seed=args.seed)
        precision, recall = precision_recall_from_cm(cm)
        print(f'Mean accuracy: {mean_acc:.4f}')
        print('Labels:', labels)
        print('Confusion matrix:\n', cm)
        for lab, pval, rval in zip(labels, precision, recall):
            print(f'  {lab}: precision={pval:.3f} recall={rval:.3f}')
        sys.exit(0)

    # If model provided load it
    clf = None
    if args.model:
        if not os.path.exists(args.model):
            print('Model file not found:', args.model)
            sys.exit(1)
        with open(args.model, 'rb') as f:
            obj = pickle.load(f)
        clf = obj.get('model', obj)
        print('Loaded model from', args.model)

    if args.split and args.split > 0:
        n = len(y)
        rng = np.random.RandomState(args.seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = int(n * (1 - args.split))
        train_idx = idx[:cut]
        test_idx = idx[cut:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        acc, labels, cm, precision, recall = evaluate_model_on_split(clf, X_train, y_train, X_test, y_test)
        print(f'Test samples: {len(y_test)}  Accuracy: {acc:.4f}')
        print('Labels:', labels)
        print('Confusion matrix:\n', cm)
        for lab, pval, rval in zip(labels, precision, recall):
            print(f'  {lab}: precision={pval:.3f} recall={rval:.3f}')
        sys.exit(0)
    else:
        # No split: evaluate on all samples using the provided model (or train+test on same data)
        if clf is None:
            print('No model provided and no split requested; run with --split or provide --model or use --kfold.')
            sys.exit(1)
        y_pred = clf.predict(X)
        labels = sorted(list(set(list(y) + list(y_pred))))
        cm = confusion_matrix(y, y_pred, labels)
        precision, recall = precision_recall_from_cm(cm)
        acc = float((np.array(y_pred) == np.array(y)).sum()) / len(y)
        print(f'Accuracy on full dataset: {acc:.4f}')
        print('Labels:', labels)
        print('Confusion matrix:\n', cm)
        for lab, pval, rval in zip(labels, precision, recall):
            print(f'  {lab}: precision={pval:.3f} recall={rval:.3f}')


if __name__ == '__main__':
    main()
