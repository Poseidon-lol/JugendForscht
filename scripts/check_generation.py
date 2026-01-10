from pathlib import Path
import pandas as pd

def main() -> None:
    hist_path = Path("experiments/active_runs/active_learning_history.csv")
    labelled_path = Path("data/processed/surrogate_labelled.csv")
    pool_path = Path("data/pool_seed.csv")

    hist = pd.read_csv(hist_path)
    labelled = pd.read_csv(labelled_path)["smiles"].dropna().unique()
    pool = pd.read_csv(pool_path)["smiles"].dropna().unique()
    seen = set(labelled) | set(pool)

    new_smiles = hist[~hist["smiles"].isin(seen)]

    print(f"History rows: {len(hist)} | unique SMILES: {hist['smiles'].nunique()}")
    print("Rows by iteration:")
    print(hist.groupby("iteration").size())
    print(f"New SMILES (not in labelled/pool): {new_smiles['smiles'].nunique()}")
    if not new_smiles.empty:
        print("Sample new SMILES:")
        print(new_smiles.head())


if __name__ == "__main__":
    main()
