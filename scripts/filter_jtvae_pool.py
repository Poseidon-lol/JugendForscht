"""
Filter the JT-VAE training CSV to avoid stalls on pathological SMILES.

Rules:
- Drop rows with missing conditioning targets.
- Drop SMILES longer than a max length (default 150 chars) to avoid huge/slow molecules.
- Optionally cap the dataset to a maximum number of rows (set MAX_ROWS to None to keep all).

Outputs a trimmed CSV at data/processed/jtvae_pool_filtered.csv.
"""

from pathlib import Path
import pandas as pd

# Configurable thresholds
# Tighten these if preprocessing stalls on pathological molecules
MAX_SMILES_LEN = 120
MAX_ROWS = 50000  # set to None to keep all
INPUT = Path("data/processed/jtvae_pool.csv")
OUTPUT = Path("data/processed/jtvae_pool_filtered.csv")
COND_COLS = ["HOMO_eV", "LUMO_eV", "IE_eV", "EA_eV"]


def main() -> None:
    df = pd.read_csv(INPUT)
    before = len(df)
    df = df.dropna(subset=COND_COLS)
    after_dropna = len(df)
    df = df[df["smiles"].astype(str).str.len() <= MAX_SMILES_LEN]
    after_len = len(df)
    if MAX_ROWS is not None and len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=0)
    df.to_csv(OUTPUT, index=False)
    print(f"Input rows: {before}")
    print(f"After dropna({COND_COLS}): {after_dropna}")
    print(f"After SMILES length <= {MAX_SMILES_LEN}: {after_len}")
    if MAX_ROWS is not None:
        print(f"Capped to {MAX_ROWS} rows; final: {len(df)}")
    else:
        print(f"Final rows: {len(df)}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
