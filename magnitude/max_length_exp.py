import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from magnitude_function import calculate_magnitude_function
from classification_utils import logreg


def run_max_length_experiment_curves(
    small_csv: str,
    lab_path:   str,
    max_lengths: list[int] = [32, 64, 128, 256],
    T: np.ndarray = np.linspace(0.005, 0.04, 50),
    metric: str = "cityblock",
    pca: bool = False,
) -> pd.DataFrame:
    """
    For each max_length, compute magnitude curves over T and classify.
    Returns DataFrame with ['max_length','auc'].
    """
    # Load data
    df    = pd.read_csv(small_csv)
    texts = df['text'].tolist()
    labs  = torch.load(lab_path).numpy()

    results = []
    # Outer progress bar for max_length sweep
    for idx, L in enumerate(tqdm(max_lengths, desc="max_length sweep", position=0)):
        # Tokenize & embed
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model     = AutoModel.from_pretrained("bert-base-uncased").eval()
        inputs    = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=L
        )
        with torch.no_grad():
            outputs = model(**inputs)
        embs = outputs.last_hidden_state  # (N, L, D)

        # Split indices for real vs generated
        real_inds = np.where(labs == 0)[0]
        gen_inds  = np.where(labs == 1)[0]

        # Compute magnitude curves with nested, positioned tqdm bars
        real_feats = []
        for x in tqdm(embs[real_inds], desc=f"Real curves L={L}", position=1, leave=False):
            real_feats.append(calculate_magnitude_function(x, metric, T))
        gen_feats = []
        for x in tqdm(embs[gen_inds], desc=f"Gen curves  L={L}", position=2, leave=False):
            gen_feats.append(calculate_magnitude_function(x, metric, T))

        real_feats = np.array(real_feats)
        gen_feats  = np.array(gen_feats)

        # Classification AUC
        auc = logreg(real_feats, gen_feats, pca=pca)
        results.append({ 'max_length': L, 'auc': auc })

    return pd.DataFrame(results)