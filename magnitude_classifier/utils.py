import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from collections import defaultdict as ddict

import pickle as pkl

def load_data(labels_path:str, dataset_path:str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    labs = torch.load(labels_path)
    dataset = pd.read_csv(dataset_path)
    return labs.numpy(), dataset



def logreg_and_find_generated_misclassifications(
        magnitude_f: np.ndarray,
        labels: np.ndarray | torch.Tensor,
        dataset: pd.DataFrame,
        pca: int | bool = False,
        n_splits: int = 6,
        **lr_kwargs
    ) -> tuple[float, ddict]:
    """
    Same training as `logreg` but also returns the list of (text, source)
    for generated samples (label=1) that were predicted as real (0).
    """
    if type(labels) == torch.Tensor:
        labels = labels.numpy()

    X = magnitude_f
    y = labels

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=21)
    aucs = []
    mistakes = ddict(int)

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)

        if pca:
            pca_model = PCA(n_components=min(pca, X_tr_s.shape[1]), random_state=21)
            X_tr_s = pca_model.fit_transform(X_tr_s)
            X_va_s = pca_model.transform(X_va_s)

        clf = LogisticRegression(**lr_kwargs).fit(X_tr_s, y_tr)
        probs = clf.predict_proba(X_va_s)[:, 1]
        preds = (probs >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_va, probs))

        # collect every generated (y_va==1) predicted as real (preds==0)
        for i, idx in enumerate(val_idx):
            if y_va[i] == 1 and preds[i] == 0:
                mistakes[dataset["source"].loc[idx]] += 1

    print(f"Mean CV AUC: {np.mean(aucs):.4f}")
    print(f"Generated→Real misclassifications: {sum(mistakes.values())}\n")

    return np.mean(aucs), mistakes