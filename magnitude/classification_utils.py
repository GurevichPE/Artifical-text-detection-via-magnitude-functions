# classification_utils.py

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def logreg(reals: np.ndarray,
           gens: np.ndarray,
           pca: bool = False,
           n_splits: int = 6) -> float:
    """
    Train logistic regression on (reals vs gens) via StratifiedKFold CV
    and return mean ROC-AUC.
    Labels: 0 = real, 1 = generated
    """
    X = np.vstack([reals, gens])
    y = np.zeros(len(reals) + len(gens), dtype=int)
    y[len(reals):] = 0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=21)
    aucs = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # scale
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)

        # optional PCA
        if pca:
            pca_model = PCA(n_components=min(700, X_tr_s.shape[1]), random_state=21)
            X_tr_s = pca_model.fit_transform(X_tr_s)
            X_va_s = pca_model.transform(X_va_s)

        # train & evaluate
        clf = LogisticRegression().fit(X_tr_s, y_tr)
        probs = clf.predict_proba(X_va_s)[:, 1]
        aucs.append(roc_auc_score(y_va, probs))

    return float(np.mean(aucs))


def logreg_and_find_generated_misclassifications(
        reals: np.ndarray,
        gens: np.ndarray,
        reals_texts: list[str],
        gens_texts:  list[str],
        reals_sources: list[str],
        gens_sources:  list[str],
        pca: bool = False,
        n_splits: int = 6
    ) -> list[tuple[str,str]]:
    """
    Same training as `logreg` but also returns the list of (text, source)
    for generated samples (label=1) that were predicted as real (0).
    """
    X = np.vstack([reals, gens])
    y = np.zeros(len(reals) + len(gens), dtype=int)
    y[len(reals):] = 0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=21)
    aucs = []
    mistakes: list[tuple[str,str]] = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)

        if pca:
            pca_model = PCA(n_components=min(700, X_tr_s.shape[1]), random_state=21)
            X_tr_s = pca_model.fit_transform(X_tr_s)
            X_va_s = pca_model.transform(X_va_s)

        clf = LogisticRegression().fit(X_tr_s, y_tr)
        probs = clf.predict_proba(X_va_s)[:, 1]
        preds = (probs >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_va, probs))

        # collect every generated (y_va==1) predicted as real (preds==0)
        for i, idx in enumerate(val_idx):
            if y_va[i] == 1 and preds[i] == 0:
                rel = idx - len(reals)
                mistakes.append((gens_texts[rel], gens_sources[rel]))

    print(f"Mean CV AUC: {np.mean(aucs):.4f}")
    print(f"Generated→Real misclassifications: {len(mistakes)}\n")
    print("Some examples:")
    for text, src in mistakes[:5]:
        print(f"[{src}] {text[:100]}…\n")

    return mistakes
