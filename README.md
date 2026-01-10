# Projektbeschreibung

Wir entwickeln eine KI-gest??tzte Entdeckungspipeline f??r organische Halbleiter: Ein JT-VAE-Generator erzeugt p- und n-dotierbare OSC-Kandidaten, ein Surrogat (MPNN, optional SchNet-3D) bewertet ihre elektronischen Eigenschaften, und eine Active-Learning-Schleife priorisiert Molek??le f??r Hochfidelit??ts-QC, Synthese und experimentelle Tests. Ziel ist es, verwertbare Materialien f??r organische Solarzellen zu identifizieren und in die Laborsynthese zu bringen.

# OSC Discovery Plan

## Phase 1 - Data Foundation
- [ ] Consolidate qm9, QMugs_curatedDFT, osc_data, csce_gap_synth into `data/raw/all_mols.csv` (include HOMO/LUMO, IE/EA, lambda_hole/lambda_electron, dipole, polarizability, packing surrogate, miscibility surrogate, synthetic accessibility, provenance tags).
- [ ] Audit coverage of each property and flag molecules with missing descriptors.
- [ ] Re-run `noteb/1_data_prep.ipynb` for both surrogate and generator configs to produce train/val/test splits, fragment vocab, normalization stats, tree adjacency tensors.
- [x] Point surrogate (`configs/train_conf.yaml`), generator (`configs/gen_conf.yaml`), and active-loop (`configs/active_learn.yaml`) to merged processed datasets.

## Phase 2 - High-Fidelity Property Pipeline
- [x] Added QC workflow modules under `src/qc/` (geometry prep, external executors, pipeline, storage).
- [x] Provided Psi4/Gaussian/ORCA command wrappers with robust error handling and heuristic fallback (`SemiEmpiricalExecutor`).
- [x] Implemented asynchronous orchestration via `AsyncQCManager` and upgraded `DFTInterface` to accept futures (submit/fetch/retry friendly).
- [x] Added `QCResultStore` helper to persist QC outputs (properties + metadata) for later merges.
- [ ] TODO: connect real QC/MD executables or cluster schedulers, extend log parsers for production outputs, and push results automatically into the master dataset.

## Phase 3 - Model Conditioning & Training
- [ ] Extend `target_columns` once high-fidelity labels (lambda, polarizability, packing score, etc.) are available.
- [ ] Retrain surrogate ensemble with enriched labels; log MAE/RMSE/NLL/CRPS, calibrate uncertainty, version checkpoints.
- [ ] Retrain JT-VAE with updated property weights and refreshed fragment vocabulary.
- [ ] Refresh active-learning seed/pool CSVs from the cleaned dataset.

## Phase 4 - Active Learning Enhancements
- [ ] Tune acquisition objectives (pareto_ucb weights), diversity heuristics, and assembly parameters.
- [ ] Enable generator refresh schedule (epochs, lr) informed by new labelled data.
- [ ] Add run management: resume checkpoints, iteration logging, and diagnostics plots.

## Phase 5 - Automation & Tooling
- [ ] Scripts/notebooks to merge QC results, trigger retraining, archive configs.
- [ ] MLflow/TensorBoard instrumentation for training and AL metrics.
- [ ] Reporting notebook (predictions -> QC -> experimental status).
- [ ] Versioned data artifacts (DVC or git-lfs).

## Phase 6 - Experimental Hand-off
- [ ] Define promotion criteria (IE/EA windows, lambda caps, stability flags, synthetic accessibility).
- [ ] Produce concise hand-off packages for synthesis teams and track experimental outcomes.

## Phase 7 - Continuous Improvement
- [ ] Add new property heads (mobility, miscibility) as data arrives.
- [ ] Explore richer acquisition/RL policies once baseline ROI flattens.
- [x] Prepare CUDA/ROCm-ready training stack (auto device selection + AMP + torch.compile toggles).
- [ ] Keep iterating: generation -> surrogate filter -> QC/MD -> experiments -> dataset update -> retrain.

Extras:
If you later set up MPI or multiple cores, bump max_workers in configs/qc_pipeline.yaml (or set ORCA's %PAL directive inside OrcaExecutor._write_input).

## CUDA / GPU Workflow
- The default configs (`configs/train_conf.yaml`, `configs/gen_conf.yaml`, `configs/active_learn.yaml`) now set `device: auto` and enable AMP + `torch.compile` so training will transparently use CUDA kernels when GPUs are available (falls back to CPU otherwise).
- CLI overrides are available: `python src/main.py train-surrogate --device cuda --amp --compile` and `python src/main.py train-generator --device cuda:0 --compile-mode max-autotune`.
- Active-learning runs accept `--device/--surrogate-device/--generator-device` to control inference placement; generator sampling reuses the same device for CUDA kernels.
- Mixed precision + compiled kernels can be disabled at runtime via `--no-amp` / `--no-compile` if debugging numerical issues.
