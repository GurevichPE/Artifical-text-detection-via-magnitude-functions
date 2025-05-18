import numpy as np
from utils import *
import pickle as pkl
from tqdm import tqdm
import torch

def main():
    SAVEDIR = "/workspace/mnt/local/data/pgurevich/magnitude"
    MAGNITUDE_PATH = "/workspace/mnt/local/data/pgurevich/magnitude"
    DATA_PATH = "/workspace/mnt/local/data/pgurevich/magnitude/small_data.csv"
    LABELS_PATH = "/workspace/mnt/local/data/pgurevich/magnitude/labels.pt"
    MODE = "both"       # 'magnitude' for magnitude function or
                        # 'embeddings' for embeddings or
                        # 'both' for concatenated embeddings and magnitudes

    LEN_GRID = [288]
    N_STEPS = 300
    
    labels, data = load_data(LABELS_PATH, DATA_PATH)
    aucs = []
    misclasses = []

    for n in tqdm(LEN_GRID):
        if MODE == 'magnitude':
            with open(f"{MAGNITUDE_PATH}/magnitude_f_{n}_{N_STEPS}_steps.pkl", 'rb') as f:
                mags = pkl.load(f)
            mags = np.array(mags)
        elif MODE == 'embeddings':
            mags = torch.load(f"{MAGNITUDE_PATH}/embeddings_{n}.pt")
            mags = mags.mean(dim=1).numpy()

        elif MODE == 'both':
            with open(f"{MAGNITUDE_PATH}/magnitude_f_{n}_{N_STEPS}_steps.pkl", 'rb') as f:
                mags = pkl.load(f)
            mags = np.array(mags)
            embs = torch.load(f"{MAGNITUDE_PATH}/embeddings_{n}.pt")
            embs = embs.mean(dim=1).numpy()
            mags = np.concatenate([embs, mags], axis=1)


        auc, mistake = logreg_and_find_generated_misclassifications(
            mags,  labels, data, pca=False, n_splits=6, max_iter=1000
        )

        aucs.append(auc)
        misclasses.append(mistake)

    final_result = {
        "aucs" : aucs,
        "mistakes" : misclasses,
        "len_grid" : LEN_GRID
    }

    with open(f"{SAVEDIR}/clf_results_{MODE}_final.pkl", "wb") as f:
        pkl.dump(final_result, f)


if __name__ == "__main__":
    main()