import pandas as pd

DATA_PATH = "/workspace/mnt/local/data/pgurevich/magnitude/train_v2_drcat_02.csv"
SAVEDIR = "/workspace/mnt/local/data/pgurevich/magnitude"
SAMPLE_SIZE = 500


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)

    # Sample 1k rows for each label
    sample_0 = data[data['label'] == 0].sample(n=SAMPLE_SIZE, random_state=42)
    sample_1 = data[data['label'] == 1].sample(n=SAMPLE_SIZE, random_state=42)

    # Combine and shuffle the samples
    subsample = pd.concat([sample_0, sample_1], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    subsample.to_csv(f"{SAVEDIR}/small_data.csv", index=False)