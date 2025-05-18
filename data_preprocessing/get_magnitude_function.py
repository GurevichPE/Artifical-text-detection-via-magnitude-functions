import torch
import numpy as np
import pickle as pkl
from joblib import Parallel, delayed
import sys
from time import time
from tqdm import tqdm

sys.path.insert(0, "..")
from magnitude.magnitude_function import calculate_magnitude_function

def main():
    EMB_PATH = "/workspace/mnt/local/data/pgurevich/magnitude"
    SAVEDIR = "/workspace/mnt/local/data/pgurevich/magnitude"
    T_MIN = 1e-8
    T_MAX = 0.09
    N_STEPS = 300
    METRIC = 'cityblock'
    N_JOBS = -1
    MAX_TOKEN_LENGTHS = [288]

    t = np.linspace(T_MIN, T_MAX, N_STEPS)
    for i in MAX_TOKEN_LENGTHS:
        embs = torch.load(f"{EMB_PATH}/embeddings_{i}.pt").numpy()
        
        # Parallel computation
        mfs = Parallel(n_jobs=N_JOBS)(
            delayed(calculate_magnitude_function)(x, METRIC, t, False) 
            for x in tqdm(embs)
        )
        
        with open(f"{SAVEDIR}/magnitude_f_{i}_{N_STEPS}_steps.pkl", 'wb+') as f:
            pkl.dump(mfs, f)

    with open(f"{SAVEDIR}/t_grid_multiple_{N_STEPS}.pkl", 'wb+') as f:
        pkl.dump(t, f)

if __name__ == "__main__":
    start = time()
    main()
    stop = time()
    print("==================================")
    print(f"The code finishes with {round((stop - start) / 60, 2)} minutes")
    print("==================================")