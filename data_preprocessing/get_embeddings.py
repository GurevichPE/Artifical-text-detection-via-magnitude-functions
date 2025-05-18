"""
dataset was taken from https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset
"""

from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd


DATA_PATH = "/workspace/mnt/local/data/pgurevich/magnitude/small_data.csv"
SAVEDIR = "/workspace/mnt/local/data/pgurevich/magnitude"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
DEVICE = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 288

def main():
    print("START")


    data = pd.read_csv(DATA_PATH)
    texts = data[TEXT_COLUMN].tolist()
    labels = data[LABEL_COLUMN].to_numpy()

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=MAX_LENGTH
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract embeddings (shape: [batch_size, sequence_length, hidden_size])
    token_embeddings = outputs.last_hidden_state.cpu()

    labels = torch.from_numpy(labels)

    print("FINISHED")
    print(f"Embeddings shape: {token_embeddings.shape}")

    torch.save(token_embeddings, f"{SAVEDIR}/embeddings_{MAX_LENGTH}.pt")
    torch.save(labels, f"{SAVEDIR}/labels_{MAX_LENGTH}.pt")


if __name__ == "__main__":
    main()


