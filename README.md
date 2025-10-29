__ToDo__ 


-Auf jeden fall datensätze

Data & Splits

Add stratified/bin‑aware splits for target columns; option to preserve label distribution (src/data/dataset.py:178).
Persist and reuse NormalizationStats per dataset; verify during load (src/data/dataset.py:72–95).
Add schema validator to assert required columns and numeric dtypes at ingest.

Preprocessing / Fragments

Cache fragment vocab + fingerprints to disk (hash by SMILES list) to avoid recomputation (src/data/jt_preprocess.py:120).
Sanitise fragments: kekulise/skip invalid, suppress RDKit noise via local logger, not global (src/data/jt_preprocess.py:85).
Parameterise JTPreprocessConfig with BRICS/retrosynthesis option and minimal frequency threshold; expose in gen config.


Hardware / Backend

Add explicit device flag to surrogate/generator CLIs (cpu|cuda|directml) and log resolved backend (src/models/ensemble.py:73, src/models/jtvae_extended.py:293).
Document ROCm/WSL path; add env check command to fail early with actionable hints.
Config / CLI UX

Add python src/main.py prepare-active --from data/processed/*_train.csv --seed 200 to auto‑create labelled_seed.csv, pool_candidates.csv, and copy fragment_vocab.json.
Validate config paths at startup; print friendly errors with fix suggestions.


Logging / Tracking / Tests

Integrate TensorBoard/MLflow: log losses, metrics, learning rates, acquisition stats, and sample molecules.
Add unit tests: featurisation, dataset splits, JT‑VAE sampling shape/validity, ensemble predict API.
Provide a tiny smoke dataset and a GitHub Action to run fast checks.


-Den ganzen code checken
