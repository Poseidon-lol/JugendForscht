# OSC Discovery Plan

## Phase 1 - Data Foundation
- [ ] Consolidate qm9, QMugs_curatedDFT, osc_data, csce_gap_synth into data/raw/all_mols.csv (include HOMO/LUMO, IE/EA, lambda_hole/lambda_electron, dipole, polarizability, packing surrogate, miscibility surrogate, synthetic accessibility, provenance tags).
- [ ] Audit coverage of each property and flag molecules with missing descriptors.
- [ ] Re-run 
oteb/1_data_prep.ipynb for both surrogate and generator configs to produce train/val/test splits, fragment vocab, normalization stats, tree adjacency tensors.
- [x] Point surrogate (configs/train_conf.yaml), generator (configs/gen_conf.yaml), and active-loop (configs/active_learn.yaml) to merged processed datasets.

## Phase 2 - High-Fidelity Property Pipeline
- [x] QC workflow operational (src/qc/): RDKit geometry ? ORCA/Psi4/Gaussian (async executor) ? property parsing + storage.
- [x] ORCA integration verified (HOMO/LUMO/gap/IE/EA + Reorganisationenergies via charge ±1 runs).
- [x] Filter tooling (configs/filter_rules.yaml, scripts/filter_qc_results.py) to keep only solar-cell candidates (HOMO/LUMO/gap/IE/EA + ?).
- [x] QC outputs logged to data/processed/qc_results.csv for downstream retraining.
- [ ] TODO: connect to cluster scheduler, extend parsers for production logs, automate append-to-labelled step.

## Phase 3 - Model Conditioning & Training
- [ ] Extend 	arget_columns once high-fidelity labels beyond HOMO/LUMO/gap/IE/EA/? are available.
- [ ] Retrain surrogate ensemble with ORCA-filtered labels; log MAE/RMSE/NLL/CRPS, calibrate uncertainty, version checkpoints.
- [ ] Retrain JT-VAE with updated property weights and refreshed fragment vocabulary.
- [ ] Refresh active-learning seed/pool CSVs from the cleaned dataset.

## Phase 4 - Active Learning Enhancements
- [ ] Tune acquisition objectives (pareto_ucb weights), diversity heuristics, and assembly parameters.
- [ ] Enable generator refresh schedule (epochs, lr) informed by new labelled data.
- [ ] Add run management: resume checkpoints, iteration logging, diagnostics plots.

## Phase 5 - Automation & Tooling
- [ ] Automation scripts/notebooks to merge QC results, trigger surrogate/JT-VAE retraining, archive configs.
- [ ] MLflow/TensorBoard instrumentation for training & AL metrics.
- [ ] Reporting notebook (predictions ? QC ? experimental status).
- [ ] Versioned data artifacts (DVC/git-lfs).

## Phase 6 - Experimental Hand-off
- [ ] Define promotion criteria (IE/EA windows, ? caps, stability flags, synthetic accessibility).
- [ ] Produce concise hand-off packages for synthesis teams and track experimental outcomes.

## Phase 7 - Continuous Improvement
- [ ] Add new property heads (mobility, miscibility) as data arrives.
- [ ] Explore richer acquisition/RL policies once baseline ROI flattens.
- [ ] Plan hardware upgrades (CUDA/ROCm) for unified GPU training.
- [ ] Keep iterating: generation ? surrogate filter ? QC/MD ? experiments ? dataset update ? retrain.
