__ToDo__ 


-Auf jeden fall datensätze

The JT‑VAE currently targets HOMO/LUMO gaps only.
The shipped datasets provide enough information to train on those gaps, but they don’t contain the broader property set you’d need for dopant efficiency (e.g. adiabatic IP/EA, reorganisation energy, polarizability, packing descriptors). If you want the generator and surrogate to learn those, you’ll need to augment the CSVs with those columns (from DFT or experiment) and update target_columns accordingly.
  -> Also jtvae target parameter expanden auf: IE, EA, λ, dipole, polarizability, maybe a packing descriptor

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




Code an sich ist lange nicht fertig:
Phase 1 – Data Foundation

Consolidate raw datasets (qm9, QMugs_curatedDFT, osc_data, csce_gap_synth) into one canonical table (data/raw/all_mols.csv) with: smiles, HOMO/LUMO, IE/EA, λe/λh, dipole, polarizability, torsional/planarity descriptors, miscibility surrogates, synthetic accessibility, provenance.
Audit missing-property coverage; tag molecules lacking key descriptors.
Re‑run noteb/1_data_prep.ipynb to produce cleaned splits (*_train/val/test.csv), fragment vocab, normalization stats, tree_adj tensors.
Update config paths (train_conf.yaml, gen_conf.yaml, active_learn.yaml) to point to the merged dataset.

Phase 2 – High-Fidelity Property Pipeline

Design QC workflow: SMILES → 3D geometry (RDKit), optimisation + single-point (DFT/TDDFT/MD) to compute IE/EA, λ, dipole, polarizability, packing surrogate, stability metrics.
Script executable wrappers (Psi4, Gaussian, ORCA, or MD) with robust output parsing and error handling.
Implement asynchronous job orchestration (FireWorks, Dask, SLURM, or local queue) to run batches.
Replace PseudoDFTSolver by wiring your executor into DFTInterface; support submit, fetch, logging, retries.
Store QC outputs and metadata (level of theory, wall time) back into the master dataset.

Phase 3 – Model Conditioning & Training

Extend dataset.target_columns in configs to include new descriptors once QC data is available.
Retrain surrogate ensemble (CPU) with the enriched labels; log MAE/RMSE/NLL/CRPS, temperature‑calibrate uncertainty; version checkpoints.
Retrain JT‑VAE (DirectML GPU) with updated joint loss weights (KL/property/adjacency); monitor training curves; save new fragment vocab.
Refresh active-learning seed/pool (data/labelled_seed.csv, data/pool_candidates.csv) from the cleaned dataset and copy fragment vocab.

Phase 4 – Active Learning Loop Enhancements

Configure acquisition (pareto_ucb/multi_ucb, weights = dopant objectives); verify diagnostics plots under experiments/active_runs/diagnostics.
Finalise diversity heuristics (Tanimoto threshold, fragment penalties, SAScore/logP filters) and beam-search parameters (beam_width, topk_per_node).
Enable generator refresh: set cadence, epochs, lr for fine-tuning on newly labelled data.
Implement run management: checkpoint history, resume support, iteration logging.

Phase 5 – Automation & Tooling

Build scripts/notebooks to merge QC results into labelled_seed.csv, trigger surrogate/JT‑VAE retraining, and archive configs.
Instrument MLflow/TensorBoard (or similar) for training metrics, acquisition scores, diversity stats; set up alerting if performance drifts.
Add reporting notebook summarising candidate pipeline (predictions → QC results → experimental status).
Version-control data artifacts (e.g., DVC or git‑LFS) for reproducibility.

Phase 6 – Experimental Hand-off

Define cut-off criteria for lab promotion (IE/EA alignment, λ caps, stability flags, synthetic accessibility).
Generate clean hand-off packages (SMILES, descriptors, predicted properties, QC values) for synthetic chemists.
Track experimental outcomes (success/failure, measured conductivity/stability) and feed back into the dataset for future retraining.

Phase 7 – Continuous Improvement / Stretch Goals

Add on-the-fly property heads (e.g., mobility, miscibility) once lab/QC data accumulates.
Explore advanced acquisition or reinforcement-learning policies if ROI plateaus.
Investigate environment upgrades (NVIDIA hardware or future ROCm support) to unify GPU training.
Keep iterating the active-learning cycle: generation → surrogate filter → QC/MD → experimental validation → dataset update → retrain.
