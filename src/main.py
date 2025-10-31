from __future__ import annotations

import sys
import argparse
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

import pandas as pd
import torch

# Ensure the project root (with src/) is on sys.path
PROJECT_ROOT = Path().resolve()
for candidate in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
    if (candidate / "src").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
else:
    raise RuntimeError("Could not locate project root containing src/")


from src.data.dataset import load_dataframe, split_dataframe
from src.models.ensemble import EnsembleConfig, SurrogateEnsemble
from src.utils.config import load_config
from src.utils.log import setup_logging

if TYPE_CHECKING:
    from src.models.jtvae_extended import JTVAE, JTVDataset


def train_surrogate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    data_cfg = cfg.dataset
    df = load_dataframe(data_cfg.path)
    target_columns = list(getattr(data_cfg, "target_columns", []))
    if not target_columns:
        raise ValueError("Config dataset.target_columns must list at least one property for surrogate training.")
    missing = [col for col in target_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Surrogate targets missing from dataframe: {missing}")
    keep_cols = ["smiles"] + target_columns
    df = df[keep_cols]
    before = len(df)
    df = df.dropna(subset=target_columns)
    dropped = before - len(df)
    if dropped > 0:
        logging.getLogger(__name__).warning(
            "Dropped %d rows with missing surrogate targets (remaining %d).", dropped, len(df)
        )
    split = split_dataframe(df, val_fraction=data_cfg.val_fraction, test_fraction=0.0, seed=args.seed)

    ens_cfg = EnsembleConfig(
        n_models=cfg.surrogate.n_models,
        epochs=cfg.surrogate.epochs,
        batch_size=cfg.surrogate.batch_size,
        lr=cfg.surrogate.lr,
        weight_decay=cfg.surrogate.weight_decay,
        patience=getattr(cfg.surrogate, "patience", max(10, cfg.surrogate.epochs // 5)),
        scheduler_patience=getattr(cfg.surrogate, "scheduler_patience", 15),
        loss=cfg.surrogate.loss,
        dropout=cfg.surrogate.dropout,
        hidden_dim=getattr(cfg.surrogate, "hidden_dim", 128),
        message_layers=getattr(cfg.surrogate, "message_layers", 3),
        readout_dim=getattr(cfg.surrogate, "readout_dim", getattr(cfg.surrogate, "hidden_dim", 128)),
        readout=getattr(cfg.surrogate, "readout", "mlp"),
        pooling=getattr(cfg.surrogate, "pooling", "mean"),
        grad_clip=getattr(cfg.surrogate, "grad_clip", 0.0),
        mc_dropout_samples=getattr(cfg.surrogate, "mc_dropout_samples", 0),
        calibrate=getattr(cfg.surrogate, "calibrate", True),
        save_dir=Path(cfg.surrogate.save_dir),
        device=getattr(cfg.surrogate, "device", None),
    )
    surrogate = SurrogateEnsemble(ens_cfg)
    surrogate.fit(split.train, split.val)
    surrogate.save_all()
    print(f"Surrogate ensemble trained and saved to {cfg.surrogate.save_dir}")


def _load_jtvae_from_ckpt(ckpt: Path, fragment_vocab_size: int, cond_dim: int) -> JTVAE:
    from src.models.jtvae_extended import JTVAE

    state = torch.load(ckpt, map_location="cpu")
    hidden_dim = state["encoder.tree_encoder.input_proj.weight"].shape[0]
    node_feat_dim = state["encoder.tree_encoder.input_proj.weight"].shape[1]
    z_dim = state["encoder.fc_mu.weight"].shape[0]
    positional_key = None
    for key in state.keys():
        if key.endswith("decoder.positional"):
            positional_key = key
            break
    max_tree_nodes = state[positional_key].shape[0] if positional_key else 12
    model = JTVAE(
        node_feat_dim=node_feat_dim,
        fragment_vocab_size=fragment_vocab_size,
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        cond_dim=cond_dim,
        max_tree_nodes=max_tree_nodes,
    )
    model.load_state_dict(state)
    return model


def train_generator(args: argparse.Namespace) -> None:
    from src.data.jt_preprocess import JTPreprocessConfig, build_fragment_vocab, prepare_jtvae_examples
    from src.models.jtvae_extended import JTVAE, JTVDataset, train_jtvae

    cfg = load_config(args.config)
    logger = logging.getLogger(__name__)
    data_cfg = cfg.dataset
    logger.info("Loading JT-VAE dataset from %s", data_cfg.path)
    df = pd.read_csv(data_cfg.path)
    logger.info("Loaded %d molecules for JT-VAE training.", len(df))
    min_count = getattr(data_cfg, "fragment_min_count", 1)
    frag2idx, idx2frag = build_fragment_vocab(df["smiles"], min_count=min_count)
    logger.info("Fragment vocabulary size: %d (min_count=%s)", len(frag2idx), min_count)
    cond_cols = cfg.dataset.target_columns
    jt_config = JTPreprocessConfig(
        max_fragments=cfg.dataset.max_fragments,
        condition_columns=cond_cols,
    )
    logger.info(
        "Preparing JT-VAE examples (max_fragments=%s, condition_columns=%s)...",
        cfg.dataset.max_fragments,
        cond_cols,
    )
    examples = prepare_jtvae_examples(
        df,
        frag2idx,
        config=jt_config,
    )
    logger.info("Prepared %d JT-VAE examples.", len(examples))
    dataset = JTVDataset(examples)
    logger.info("Constructed JTVDataset with %d entries.", len(dataset))
    cond_dim_value = len(cond_cols) if getattr(cfg.model, "cond_dim", None) is None else cfg.model.cond_dim
    if cond_dim_value != len(cond_cols):
        logger.warning(
            "Configured cond_dim (%s) differs from number of condition columns (%s). "
            "Using %s for property head.",
            cond_dim_value,
            len(cond_cols),
            len(cond_cols),
        )
        cond_dim_value = len(cond_cols)
    tree_feat_dim = examples[0]["tree_x"].size(1)
    graph_feat_dim = examples[0]["graph_x"].size(1)
    logger.info("Tree feature dim: %s | Graph feature dim: %s", tree_feat_dim, graph_feat_dim)
    model = JTVAE(
        tree_feat_dim=tree_feat_dim,
        graph_feat_dim=graph_feat_dim,
        fragment_vocab_size=len(frag2idx),
        z_dim=cfg.model.z_dim,
        hidden_dim=cfg.model.hidden_dim,
        cond_dim=cond_dim_value,
        max_tree_nodes=cfg.dataset.max_fragments,
    )
    kl_weight = getattr(cfg.training, "kl_weight", 0.5)
    property_weight = getattr(cfg.training, "property_loss_weight", 0.0)
    adjacency_weight = getattr(cfg.training, "adjacency_loss_weight", 1.0)
    logger.info(
        "Starting JT-VAE training: epochs=%s batch_size=%s lr=%s (kl=%s, prop=%s, adj=%s)",
        cfg.training.epochs,
        cfg.training.batch_size,
        cfg.training.lr,
        kl_weight,
        property_weight,
        adjacency_weight,
    )
    train_jtvae(
        model,
        dataset,
        frag2idx,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        save_dir=cfg.save_dir,
        kl_weight=kl_weight,
        property_weight=property_weight,
        adj_weight=adjacency_weight,
    )
    vocab_path = Path(cfg.save_dir) / "fragment_vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(idx2frag, f, indent=2)
    if jt_config.condition_stats:
        stats_path = Path(cfg.save_dir) / "condition_stats.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(jt_config.condition_stats, f, indent=2)
    print(f"Generator checkpoints and vocab saved to {cfg.save_dir}")


def run_active_loop(args: argparse.Namespace) -> None:
    from src.active_learn.acq import AcquisitionConfig
    from src.active_learn.loop import ActiveLearningLoop, LoopConfig
    from src.active_learn.sched import SchedulerConfig
    from src.models.jtvae_extended import JTVAE

    cfg = load_config(args.config)
    acq_cfg = AcquisitionConfig(**cfg.acquisition)
    sched_cfg = SchedulerConfig(**cfg.scheduler)
    loop_cfg = LoopConfig(
        batch_size=cfg.loop.batch_size,
        acquisition=acq_cfg,
        scheduler=sched_cfg,
        target_columns=tuple(cfg.loop.target_columns),
        maximise=tuple(cfg.loop.maximise),
        generator_samples=cfg.loop.generator_samples,
        results_dir=Path(cfg.loop.results_dir),
        assemble=dict(getattr(cfg, "assemble", {})),
        diversity_threshold=float(getattr(cfg.loop, "diversity_threshold", 0.0)),
        diversity_metric=getattr(cfg.loop, "diversity_metric", "tanimoto"),
        generator_refresh=dict(getattr(cfg.loop, "generator_refresh", {})),
        property_aliases=dict(getattr(cfg.loop, "property_aliases", {})),
    )

    labelled = pd.read_csv(cfg.data.labelled)
    pool = pd.read_csv(cfg.data.pool)
    surrogate = SurrogateEnsemble.from_directory(Path(args.surrogate_dir))

    generator = None
    fragment_vocab: Optional[Dict[str, int]] = None
    if args.generator_ckpt and cfg.data.fragment_vocab:
        vocab_path = Path(cfg.data.fragment_vocab)
        if vocab_path.exists():
            with vocab_path.open("r", encoding="utf-8") as f:
                fragment_vocab = {k: int(v) for k, v in json.load(f).items()}
        if fragment_vocab:
            generator = _load_jtvae_from_ckpt(Path(args.generator_ckpt), len(fragment_vocab), cond_dim=len(loop_cfg.target_columns))

    dft = None
    qc_store = None
    qc_manager = None
    if args.use_pseudo_dft:
        from src.data.dft_int import DFTInterface  # lazy import

        dft = DFTInterface()
    else:
        qc_cfg = getattr(cfg, "qc", None)
        if qc_cfg:
            from src.data.dft_int import DFTInterface  # lazy import
            from src.qc.config import GeometryConfig, QuantumTaskConfig, PipelineConfig
            from src.qc.pipeline import QCPipeline, AsyncQCManager
            from src.qc.storage import QCResultStore

            pipeline_data = {}
            pipeline_config_path = getattr(qc_cfg, "pipeline_config", None)
            if pipeline_config_path:
                pipeline_path = Path(pipeline_config_path)
                if not pipeline_path.exists():
                    raise FileNotFoundError(f"QC pipeline config not found: {pipeline_path}")
                with pipeline_path.open("r", encoding="utf-8") as fh:
                    pipeline_data = yaml.safe_load(fh) or {}
            defaults = PipelineConfig()
            geometry_cfg = GeometryConfig(**pipeline_data.get("geometry", {}))
            quantum_cfg = QuantumTaskConfig(**pipeline_data.get("quantum", {}))
            pipeline_kwargs = {
                "geometry": geometry_cfg,
                "quantum": quantum_cfg,
                "work_dir": Path(pipeline_data.get("work_dir", defaults.work_dir)),
                "max_workers": pipeline_data.get("max_workers", defaults.max_workers),
                "poll_interval": pipeline_data.get("poll_interval", defaults.poll_interval),
                "cleanup_workdir": pipeline_data.get("cleanup_workdir", defaults.cleanup_workdir),
                "store_metadata": pipeline_data.get("store_metadata", defaults.store_metadata),
                "tracked_properties": tuple(pipeline_data.get("tracked_properties", defaults.tracked_properties)),
            }
            pipeline_config = PipelineConfig(**pipeline_kwargs)
            if hasattr(qc_cfg, "engine"):
                pipeline_config.quantum.engine = qc_cfg.engine
            if hasattr(qc_cfg, "method"):
                pipeline_config.quantum.method = qc_cfg.method
            if hasattr(qc_cfg, "basis"):
                pipeline_config.quantum.basis = qc_cfg.basis
            if hasattr(qc_cfg, "properties"):
                pipeline_config.quantum.properties = tuple(qc_cfg.properties)
            if isinstance(pipeline_config.work_dir, str):
                pipeline_config.work_dir = Path(pipeline_config.work_dir)
            if isinstance(pipeline_config.quantum.scratch_dir, str):
                pipeline_config.quantum.scratch_dir = Path(pipeline_config.quantum.scratch_dir)
            pipeline_config.quantum.properties = tuple(pipeline_config.quantum.properties)
            pipeline_config.tracked_properties = tuple(pipeline_config.tracked_properties)
            store_path = getattr(qc_cfg, "result_store", None)
            if store_path:
                qc_store = QCResultStore(Path(store_path))
            pipeline = QCPipeline(pipeline_config, result_store=qc_store)
            qc_manager = AsyncQCManager(pipeline, max_workers=pipeline_config.max_workers)
            dft = DFTInterface(executor=qc_manager)

    loop = ActiveLearningLoop(
        surrogate=surrogate,
        labelled=labelled,
        pool=pool,
        config=loop_cfg,
        generator=generator,
        fragment_vocab=fragment_vocab,
        dft=dft,
    )
    loop.run(args.iterations)
    loop.save_history()
    if qc_manager is not None:
        qc_manager.shutdown()
    print("Active learning completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="OSC discovery toolkit")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train-surrogate")
    train_parser.add_argument("--config", default="configs/train_conf.yaml")
    train_parser.add_argument("--seed", type=int, default=42)

    gen_parser = subparsers.add_parser("train-generator")
    gen_parser.add_argument("--config", default="configs/gen_conf.yaml")

    al_parser = subparsers.add_parser("active-loop")
    al_parser.add_argument("--config", default="configs/active_learn.yaml")
    al_parser.add_argument("--surrogate-dir", default="models/surrogate")
    al_parser.add_argument("--generator-ckpt", default=None)
    al_parser.add_argument("--iterations", type=int, default=5)
    al_parser.add_argument("--use-pseudo-dft", action="store_true")

    args = parser.parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level=log_level)

    if args.command == "train-surrogate":
        train_surrogate(args)
    elif args.command == "train-generator":
        train_generator(args)
    elif args.command == "active-loop":
        run_active_loop(args)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
