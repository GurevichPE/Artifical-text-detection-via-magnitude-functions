import torch
import numpy as np
import pickle as pkl
from joblib import Parallel, delayed
import sys
from time import time

sys.path.insert(0, "..")
from magnitude.magnitude_function import calculate_magnitude_function

def main():
    EMB_PATH = "/workspace/mnt/local/data/pgurevich/magnitude/embeddings.pt"
    SAVEDIR = "/workspace/mnt/local/data/pgurevich/magnitude"
    T_MIN = 0.001
    T_MAX = 0.1
    N_STEPS = 500
    METRIC = 'cityblock'
    N_JOBS = -1

    embs = torch.load(EMB_PATH).numpy()
    t = np.linspace(T_MIN, T_MAX, N_STEPS)
    
    # Parallel computation
    mfs = Parallel(n_jobs=N_JOBS)(
        delayed(calculate_magnitude_function)(x, METRIC, t) 
        for x in embs
    )
    
    with open(f"{SAVEDIR}/magnitude_f.pkl", 'wb+') as f:
        pkl.dump(mfs, f)

    with open(f"{SAVEDIR}/t_grid.pkl", 'wb+') as f:
        pkl.dump(t, f)

if __name__ == "__main__":
    start = time()
    main()
    stop = time()
    print("==================================")
    print(f"The code finishes with {round((stop - start) / 60, 2)} minutes")
    print("==================================")