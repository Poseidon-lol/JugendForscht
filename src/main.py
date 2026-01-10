from __future__ import annotations

import sys
import re
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
from src.utils.device import get_device
from src.utils.log import setup_logging

if TYPE_CHECKING:
    from src.models.jtvae_extended import JTVAE, JTVDataset
    from src.models.schnet_full import RealSchNetModel


def train_surrogate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.device:
        cfg.surrogate.device = args.device
    if args.amp is not None:
        cfg.surrogate.use_amp = bool(args.amp)
    if args.compile is not None:
        cfg.surrogate.compile = bool(args.compile)
    if args.compile_mode:
        cfg.surrogate.compile_mode = args.compile_mode
    if args.compile_fullgraph is not None:
        cfg.surrogate.compile_fullgraph = bool(args.compile_fullgraph)
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
        use_amp=bool(getattr(cfg.surrogate, "use_amp", False)),
        compile=bool(getattr(cfg.surrogate, "compile", False)),
        compile_mode=getattr(cfg.surrogate, "compile_mode", "default"),
        compile_fullgraph=bool(getattr(cfg.surrogate, "compile_fullgraph", False)),
    )
    surrogate = SurrogateEnsemble(ens_cfg)
    surrogate.fit(split.train, split.val)
    surrogate.save_all()
    print(f"Surrogate ensemble trained and saved to {cfg.surrogate.save_dir}")

def train_surrogate_3d(args: argparse.Namespace) -> None:
    """Train a SchNet-based 3D surrogate on MolBlock geometries."""
    import yaml
    import pandas as pd
    from src.data.featurization_3d import dataframe_to_3d_dataset
    from src.models.schnet_surrogate import SchNetConfig, train_schnet

    cfg = load_config(args.config)
    if getattr(args, "device", None):
        cfg.training.device = args.device
    data_cfg = cfg.dataset
    df = pd.read_csv(data_cfg.path)
    target_columns = list(getattr(data_cfg, "target_columns", []))
    if not target_columns:
        raise ValueError("Config dataset.target_columns must list at least one property for surrogate training.")
    mol_col = getattr(data_cfg, "mol_column", "mol")
    smi_col = getattr(data_cfg, "smiles_column", getattr(data_cfg, "smile_column", "smile"))
    ds = dataframe_to_3d_dataset(df, mol_col=mol_col, smiles_col=smi_col, target_cols=target_columns)
    if len(ds) == 0:
        raise ValueError("No valid 3D entries parsed from dataset; check mol_column/smiles_column.")
    # split
    val_fraction = float(getattr(data_cfg, "val_fraction", 0.1))
    n_val = int(len(ds) * val_fraction)
    train_ds = ds[n_val:]
    val_ds = ds[:n_val] if n_val > 0 else None
    train_cfg = cfg.training
    model_cfg = cfg.model
    sch_cfg = SchNetConfig(
        hidden_channels=model_cfg.hidden_channels,
        num_filters=model_cfg.num_filters,
        num_interactions=model_cfg.num_interactions,
        num_gaussians=model_cfg.num_gaussians,
        cutoff=model_cfg.cutoff,
        readout=model_cfg.readout,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        batch_size=train_cfg.batch_size,
        epochs=train_cfg.epochs,
        patience=train_cfg.patience,
        device=getattr(train_cfg, "device", "cpu"),
        save_dir=Path(train_cfg.save_dir),
    )
    model, hist = train_schnet(train_ds, val_ds, target_dim=len(target_columns), config=sch_cfg)
    print(f"3D surrogate trained; best model saved to {sch_cfg.save_dir/'schnet.pt'}")


def train_surrogate_3d_full(args: argparse.Namespace) -> None:
    """Train a full SchNet (PyG) surrogate on MolBlock geometries."""
    import pandas as pd
    from src.data.featurization_3d import dataframe_to_3d_dataset
    from src.models.schnet_full import RealSchNetConfig, train_schnet_full

    cfg = load_config(args.config)
    if getattr(args, "device", None):
        cfg.training.device = args.device
    data_cfg = cfg.dataset
    df = pd.read_csv(data_cfg.path)
    target_columns = list(getattr(data_cfg, "target_columns", []))
    if not target_columns:
        raise ValueError("Config dataset.target_columns must list at least one property for surrogate training.")
    mol_col = getattr(data_cfg, "mol_column", "mol")
    smi_col = getattr(data_cfg, "smiles_column", getattr(data_cfg, "smile_column", "smile"))
    ds = dataframe_to_3d_dataset(df, mol_col=mol_col, smiles_col=smi_col, target_cols=target_columns)
    if len(ds) == 0:
        raise ValueError("No valid 3D entries parsed from dataset; check mol_column/smiles_column.")
    # split
    val_fraction = float(getattr(data_cfg, "val_fraction", 0.1))
    n_val = int(len(ds) * val_fraction)
    train_ds = ds[n_val:]
    val_ds = ds[:n_val] if n_val > 0 else None
    train_cfg = cfg.training
    model_cfg = cfg.model
    sch_cfg = RealSchNetConfig(
        hidden_channels=model_cfg.hidden_channels,
        num_filters=model_cfg.num_filters,
        num_interactions=model_cfg.num_interactions,
        num_gaussians=model_cfg.num_gaussians,
        cutoff=model_cfg.cutoff,
        readout=model_cfg.readout,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        batch_size=train_cfg.batch_size,
        epochs=train_cfg.epochs,
        patience=train_cfg.patience,
        device=getattr(train_cfg, "device", "cpu"),
        save_dir=Path(train_cfg.save_dir),
    )
    model, hist = train_schnet_full(train_ds, val_ds, target_dim=len(target_columns), config=sch_cfg)
    print(f"Full SchNet surrogate trained; best model saved to {sch_cfg.save_dir/'schnet_full.pt'}")

def _load_jtvae_from_ckpt(ckpt: Path, fragment_vocab_size: int, cond_dim: int) -> JTVAE:
    from src.models.jtvae_extended import JTVAE

    state = torch.load(ckpt, map_location="cpu")
    hidden_dim = state["encoder.tree_encoder.input_proj.weight"].shape[0]
    node_feat_dim = state["encoder.tree_encoder.input_proj.weight"].shape[1]
    graph_feat_dim = state["encoder.graph_encoder.input_proj.weight"].shape[1]
    z_dim = state["encoder.fc_mu.weight"].shape[0]
    # Prefer cond_dim inferred from checkpoint (property head output or fc_mu input size)
    cond_dim_from_state = None
    if "property_head.2.weight" in state:
        cond_dim_from_state = state["property_head.2.weight"].shape[0]
    else:
        fused_dim = state["encoder.fc_mu.weight"].shape[1]
        cond_dim_from_state = max(fused_dim - 2 * hidden_dim, 0)
    if cond_dim_from_state is not None and cond_dim_from_state != cond_dim:
        logging.getLogger(__name__).info(
            "Adjusting cond_dim from %s to %s based on checkpoint.", cond_dim, cond_dim_from_state
        )
        cond_dim = cond_dim_from_state
    positional_key = None
    for key in state.keys():
        if key.endswith("decoder.positional"):
            positional_key = key
            break
    max_tree_nodes = state[positional_key].shape[0] if positional_key else 12
    model = JTVAE(
        tree_feat_dim=node_feat_dim,
        graph_feat_dim=graph_feat_dim,
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
    if args.device:
        cfg.training.device = args.device
    if args.amp is not None:
        cfg.training.use_amp = bool(args.amp)
    if args.compile is not None:
        cfg.training.compile = bool(args.compile)
    if args.compile_mode:
        cfg.training.compile_mode = args.compile_mode
    if args.compile_fullgraph is not None:
        cfg.training.compile_fullgraph = bool(args.compile_fullgraph)
    logger = logging.getLogger(__name__)
    data_cfg = cfg.dataset
    logger.info("Loading JT-VAE dataset from %s", data_cfg.path)
    df = pd.read_csv(data_cfg.path)
    logger.info("Loaded %d molecules for JT-VAE training.", len(df))
    if "smiles" not in df.columns:
        if "smile" in df.columns:
            df = df.rename(columns={"smile": "smiles"})
            logger.info("Renamed 'smile' column to 'smiles' for JT-VAE training.")
        else:
            raise KeyError("JT-VAE dataset must contain a 'smiles' (or 'smile') column.")
    min_count = getattr(data_cfg, "fragment_min_count", 1)
    fragment_method = getattr(data_cfg, "fragment_method", "ring_scaffold")
    min_fragment_heavy_atoms = getattr(data_cfg, "min_fragment_heavy_atoms", 1)
    frag2idx, idx2frag = build_fragment_vocab(
        df["smiles"],
        min_count=min_count,
        fragment_method=fragment_method,
        min_fragment_heavy_atoms=min_fragment_heavy_atoms,
    )
    if len(frag2idx) == 0 and min_count > 1:
        logger.warning(
            "Fragment vocabulary is empty with min_count=%s; lowering to 1 and rebuilding.", min_count
        )
        min_count = 1
        frag2idx, idx2frag = build_fragment_vocab(
            df["smiles"],
            min_count=min_count,
            fragment_method=fragment_method,
            min_fragment_heavy_atoms=min_fragment_heavy_atoms,
        )
    if len(frag2idx) == 0:
        raise ValueError(
            "Fragment vocabulary is empty. Provide more data or lower dataset.fragment_min_count."
        )
    logger.info(
        "Fragment vocabulary size: %d (min_count=%s, method=%s, min_heavy=%s)",
        len(frag2idx),
        min_count,
        fragment_method,
        min_fragment_heavy_atoms,
    )
    cond_cols = cfg.dataset.target_columns
    jt_config = JTPreprocessConfig(
        max_fragments=cfg.dataset.max_fragments,
        condition_columns=cond_cols,
        fragment_method=fragment_method,
        min_fragment_heavy_atoms=min_fragment_heavy_atoms,
    )
    max_heavy_atoms = getattr(data_cfg, "max_heavy_atoms", 80)
    logger.info(
        "Preparing JT-VAE examples (max_fragments=%s, condition_columns=%s, max_heavy_atoms=%s)...",
        cfg.dataset.max_fragments,
        cond_cols,
        max_heavy_atoms,
    )
    examples = prepare_jtvae_examples(
        df,
        frag2idx,
        config=jt_config,
        max_heavy_atoms=max_heavy_atoms,
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
    resume_ckpt = getattr(cfg.training, "resume_ckpt", None)
    if resume_ckpt:
        ckpt_path = Path(resume_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"JT-VAE resume checkpoint not found: {ckpt_path}")
        logger.info("Loading JT-VAE checkpoint from %s", ckpt_path)
        model = _load_jtvae_from_ckpt(ckpt_path, len(frag2idx), cond_dim_value)
    else:
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
    device_override = getattr(cfg.training, "device", None)
    use_amp = bool(getattr(cfg.training, "use_amp", False))
    compile_flag = bool(getattr(cfg.training, "compile", False))
    compile_mode = getattr(cfg.training, "compile_mode", "default")
    compile_fullgraph = bool(getattr(cfg.training, "compile_fullgraph", False))
    max_grad_norm = getattr(cfg.training, "max_grad_norm", None)
    resume_epoch = getattr(cfg.training, "resume_epoch", None)
    if resume_epoch is None and resume_ckpt:
        match = re.search(r"epoch_(\d+)", str(resume_ckpt))
        if match:
            resume_epoch = int(match.group(1))
    start_epoch = 1
    if resume_epoch is not None:
        start_epoch = max(1, int(resume_epoch) + 1)
    logger.info(
        "Starting JT-VAE training: epochs=%s start_epoch=%s batch_size=%s lr=%s (kl=%s, prop=%s, adj=%s) device=%s amp=%s compile=%s max_grad_norm=%s",
        cfg.training.epochs,
        start_epoch,
        cfg.training.batch_size,
        cfg.training.lr,
        kl_weight,
        property_weight,
        adjacency_weight,
        device_override or "auto",
        use_amp,
        compile_flag,
        max_grad_norm,
    )
    train_jtvae(
        model,
        dataset,
        frag2idx,
        device=device_override,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        save_dir=cfg.save_dir,
        kl_weight=kl_weight,
        property_weight=property_weight,
        adj_weight=adjacency_weight,
        use_amp=use_amp,
        compile=compile_flag,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        max_grad_norm=max_grad_norm,
        start_epoch=start_epoch,
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


def train_generator_3d(args: argparse.Namespace) -> None:
    """Train the 3D VAE generator on opv_db MolBlocks."""
    import pandas as pd
    from src.data.featurization_3d_gen import build_gen3d_dataset
    from src.models.vae3d import VAE3DConfig, train_vae3d

    cfg = load_config(args.config)
    data_cfg = cfg.dataset
    train_cfg = cfg.training
    model_cfg = cfg.model

    df = pd.read_csv(data_cfg.path)
    max_atoms = int(getattr(data_cfg, "max_atoms", 100))
    dataset = build_gen3d_dataset(
        df,
        mol_col=getattr(data_cfg, "mol_column", "mol"),
        smiles_col=getattr(data_cfg, "smiles_column", getattr(data_cfg, "smile_column", "smile")),
        max_atoms=max_atoms,
    )
    device = getattr(train_cfg, "device", "cpu")
    config = VAE3DConfig(
        max_atoms=max_atoms,
        z_dim=model_cfg.z_dim,
        hidden_dim=model_cfg.hidden_dim,
        lr=train_cfg.lr,
        batch_size=train_cfg.batch_size,
        epochs=train_cfg.epochs,
        patience=train_cfg.patience,
        device=device,
        save_path=train_cfg.save_path,
    )
    train_vae3d(dataset, config)
    print(f"3D generator trained and saved to {config.save_path}")


def run_active_loop(args: argparse.Namespace) -> None:
    from src.active_learn.acq import AcquisitionConfig
    from src.active_learn.loop import ActiveLearningLoop, LoopConfig
    from src.active_learn.sched import SchedulerConfig
    from src.models.jtvae_extended import JTVAE
    import pandas as pd
    from torch_geometric.loader import DataLoader

    cfg = load_config(args.config)
    acq_cfg = AcquisitionConfig(**cfg.acquisition)
    sched_cfg = SchedulerConfig(**cfg.scheduler)
    loop_device_cfg = getattr(cfg.loop, "device", None)
    surrogate_device = getattr(cfg.loop, "surrogate_device", loop_device_cfg)
    generator_device = getattr(cfg.loop, "generator_device", loop_device_cfg)
    if getattr(args, "device", None):
        surrogate_device = generator_device = args.device
    if getattr(args, "surrogate_device", None):
        surrogate_device = args.surrogate_device
    if getattr(args, "generator_device", None):
        generator_device = args.generator_device
    loop_cfg = LoopConfig(
        batch_size=cfg.loop.batch_size,
        acquisition=acq_cfg,
        scheduler=sched_cfg,
        target_columns=tuple(cfg.loop.target_columns),
        maximise=tuple(cfg.loop.maximise),
        generator_samples=cfg.loop.generator_samples,
        generator_attempts=int(getattr(cfg.loop, "generator_attempts", LoopConfig.generator_attempts)),
        results_dir=Path(cfg.loop.results_dir),
        assemble=dict(getattr(cfg, "assemble", {})),
        diversity_threshold=float(getattr(cfg.loop, "diversity_threshold", 0.0)),
        diversity_metric=getattr(cfg.loop, "diversity_metric", "tanimoto"),
        generator_refresh=dict(getattr(cfg.loop, "generator_refresh", {})),
        property_aliases=dict(getattr(cfg.loop, "property_aliases", {})),
        max_pool_eval=getattr(cfg.loop, "max_pool_eval", None),
        max_generated_heavy_atoms=getattr(cfg.loop, "max_generated_heavy_atoms", None),
        max_generated_smiles_len=getattr(cfg.loop, "max_generated_smiles_len", None),
        generated_smiles_len_factor=(
            None
            if getattr(cfg.loop, "generated_smiles_len_factor", LoopConfig.generated_smiles_len_factor) is None
            else float(getattr(cfg.loop, "generated_smiles_len_factor", LoopConfig.generated_smiles_len_factor))
        ),
        exclude_smiles_paths=tuple(getattr(cfg.loop, "exclude_smiles_paths", ())),
        min_pi_conjugated_fraction=getattr(cfg.loop, "min_pi_conjugated_fraction", None),
        require_conjugation=bool(getattr(cfg.loop, "require_conjugation", LoopConfig.require_conjugation)),
        min_conjugated_bonds=int(
            getattr(cfg.loop, "min_conjugated_bonds", LoopConfig.min_conjugated_bonds)
        ),
        min_alternating_conjugated_bonds=int(
            getattr(
                cfg.loop,
                "min_alternating_conjugated_bonds",
                LoopConfig.min_alternating_conjugated_bonds,
            )
        ),
        min_aromatic_rings=int(getattr(cfg.loop, "min_aromatic_rings", LoopConfig.min_aromatic_rings)),
        max_rotatable_bonds=getattr(cfg.loop, "max_rotatable_bonds", LoopConfig.max_rotatable_bonds),
        max_branch_points=getattr(cfg.loop, "max_branch_points", None),
        property_filters=dict(getattr(cfg.loop, "property_filters", {})),
        require_neutral=bool(getattr(cfg.loop, "require_neutral", True)),
        sa_score_max=getattr(cfg.loop, "sa_score_max", None),
        physchem_filters=dict(getattr(cfg.loop, "physchem_filters", {})),
        scaffold_unique=bool(getattr(cfg.loop, "scaffold_unique", False)),
    )

    labelled = pd.read_csv(cfg.data.labelled)
    pool = pd.read_csv(cfg.data.pool)

    def _ensure_smiles(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if "smiles" not in df.columns:
            if "smile" in df.columns:
                df = df.rename(columns={"smile": "smiles"})
            else:
                raise KeyError(f"{name} dataframe must contain a 'smiles' column (or 'smile').")
        return df

    labelled = _ensure_smiles(labelled, "labelled")
    pool = _ensure_smiles(pool, "pool")

    surrogate_device_runtime = surrogate_device
    surrogate = None
    surrogate_path = Path(args.surrogate_dir)
    dft_job_defaults: Dict[str, object] = {}

    class SchNetSurrogateWrapper:
        def __init__(self, model, device: str, target_columns, train_params=None):
            import numpy as np

            if isinstance(model, (list, tuple)):
                self.models = list(model)
            else:
                self.models = [model]
            self.device = torch.device(device)
            self.target_columns = list(target_columns)
            self.batch_size = cfg.loop.batch_size
            self.is_schnet = True
            self.train_params = train_params or {}

            # move models to device
            moved = []
            for m in self.models:
                moved.append(m.to(self.device))
            self.models = moved

        def predict(self, graphs, batch_size=None, mc_samples: int = 1, **kwargs):
            import numpy as np

            if not isinstance(graphs, (list, tuple)):
                graphs = [graphs]
            loader = DataLoader(graphs, batch_size=batch_size or self.batch_size, shuffle=False)
            mc = max(1, int(mc_samples))
            all_passes = []
            for model in self.models:
                is_training = model.training
                try:
                    model.train(mc > 1)
                    with torch.no_grad():
                        for _ in range(mc):
                            preds = []
                            for batch in loader:
                                batch = batch.to(self.device)
                                preds.append(model(batch.z, batch.pos, getattr(batch, "batch", None)).detach().cpu())
                            all_passes.append(torch.cat(preds, dim=0))
                finally:
                    if not is_training:
                        model.eval()
            stacked = torch.stack(all_passes, dim=0)  # [n_passes, N, D]
            mean = stacked.mean(dim=0).numpy()
            std = stacked.std(dim=0, unbiased=True).numpy() if stacked.shape[0] > 1 else np.zeros_like(mean)
            return mean, std, None

        def fit(self, train_df, val_df=None):
            from src.data.featurization_3d import dataframe_to_3d_dataset
            from src.models.schnet_surrogate import SchNetConfig, train_schnet
            import math

            target_cols = [c for c in self.target_columns if c in train_df.columns]
            ds_train = dataframe_to_3d_dataset(train_df, mol_col="mol", smiles_col="smiles", target_cols=target_cols)
            ds_val = None
            if val_df is not None and len(val_df) > 0:
                ds_val = dataframe_to_3d_dataset(val_df, mol_col="mol", smiles_col="smiles", target_cols=target_cols)
            if len(ds_train) == 0:
                logger.warning("SchNet retrain skipped: no valid 3D entries in training split.")
                return

            # Determine if we train pseudo or full SchNet
            is_full = any(m.__class__.__name__ == "RealSchNetModel" for m in self.models)
            n_models = int(self.train_params.get("n_models", len(self.models)))
            base_cfg = None
            if hasattr(self.models[0], "cfg"):
                base_cfg = getattr(self.models[0], "cfg")

            new_models = []
            for idx in range(max(1, n_models)):
                if is_full:
                    from src.models.schnet_full import RealSchNetConfig, train_schnet_full

                    params = {
                        "hidden_channels": getattr(base_cfg, "hidden_channels", 128) if base_cfg else 128,
                        "num_filters": getattr(base_cfg, "num_filters", 128) if base_cfg else 128,
                        "num_interactions": getattr(base_cfg, "num_interactions", 6) if base_cfg else 6,
                        "num_gaussians": getattr(base_cfg, "num_gaussians", 50) if base_cfg else 50,
                        "cutoff": getattr(base_cfg, "cutoff", 10.0) if base_cfg else 10.0,
                        "readout": getattr(base_cfg, "readout", "add") if base_cfg else "add",
                        "lr": self.train_params.get("lr", 1e-3),
                        "weight_decay": self.train_params.get("weight_decay", 0.0),
                        "batch_size": self.train_params.get("batch_size", 16),
                        "epochs": self.train_params.get("epochs", 30),
                        "patience": self.train_params.get("patience", 5),
                        "device": str(self.device),
                        "save_dir": Path(self.train_params.get("save_dir", getattr(base_cfg, "save_dir", "models/surrogate_3d_full"))),
                    }
                    sch_cfg = RealSchNetConfig(**params)
                    new_model, _ = train_schnet_full(
                        ds_train,
                        ds_val,
                        target_dim=len(target_cols),
                        config=sch_cfg,
                        save_path=Path(sch_cfg.save_dir) / f"schnet_full_member_{idx:02d}.pt",
                        seed=int(self.train_params.get("seed", 1337)) + idx,
                    )
                else:
                    params = {
                        "hidden_channels": getattr(base_cfg, "hidden_channels", 128) if base_cfg else 128,
                        "lr": self.train_params.get("lr", 1e-3),
                        "weight_decay": self.train_params.get("weight_decay", 1e-4),
                        "batch_size": self.train_params.get("batch_size", 32),
                        "epochs": self.train_params.get("epochs", 20),
                        "patience": self.train_params.get("patience", 5),
                        "device": str(self.device),
                        "save_dir": Path(self.train_params.get("save_dir", getattr(base_cfg, "save_dir", "models/surrogate_3d"))),
                    }
                    sch_cfg = SchNetConfig(**params)
                    new_model, _ = train_schnet(ds_train, ds_val, target_dim=len(target_cols), config=sch_cfg)
                new_models.append(new_model.to(self.device))
            self.models = new_models

    if surrogate_path.is_file():
        # distinguish between pseudo-SchNet and full SchNet checkpoints
        if "schnet_full" in surrogate_path.stem or surrogate_path.name == "schnet_full.pt":
            from src.models.schnet_full import RealSchNetConfig, load_schnet_full

            device_spec = get_device(surrogate_device or loop_device_cfg or "cpu")
            surrogate_device_runtime = (
                f"{device_spec.type}:{device_spec.index}" if device_spec.index is not None else device_spec.type
            )
            sch_cfg = RealSchNetConfig(device=surrogate_device_runtime)
            model = load_schnet_full(surrogate_path, target_dim=len(loop_cfg.target_columns), cfg=sch_cfg)
            surrogate = SchNetSurrogateWrapper([model], surrogate_device_runtime, loop_cfg.target_columns)
            surrogate.is_schnet_full = True  # marker
        else:
            from src.models.schnet_surrogate import SchNetConfig, load_schnet

            device_spec = get_device(surrogate_device or loop_device_cfg or "cpu")
            surrogate_device_runtime = (
                f"{device_spec.type}:{device_spec.index}" if device_spec.index is not None else device_spec.type
            )
            sch_cfg = SchNetConfig(device=surrogate_device_runtime)
            model = load_schnet(surrogate_path, target_dim=len(loop_cfg.target_columns), cfg=sch_cfg)
            surrogate = SchNetSurrogateWrapper([model], surrogate_device_runtime, loop_cfg.target_columns)
    else:
        # Directory path: check for full SchNet checkpoint inside; else fall back to ensemble
        maybe_full = Path(args.surrogate_dir) / "schnet_full.pt"
        if maybe_full.exists():
            from src.models.schnet_full import RealSchNetConfig, load_schnet_full

            device_spec = get_device(surrogate_device or loop_device_cfg or "cpu")
            surrogate_device_runtime = (
                f"{device_spec.type}:{device_spec.index}" if device_spec.index is not None else device_spec.type
            )
            sch_cfg = RealSchNetConfig(device=surrogate_device_runtime)
            model = load_schnet_full(maybe_full, target_dim=len(loop_cfg.target_columns), cfg=sch_cfg)
            surrogate = SchNetSurrogateWrapper([model], surrogate_device_runtime, loop_cfg.target_columns)
            surrogate.is_schnet_full = True
        else:
            # check for ensemble of full SchNet members
            members = sorted(Path(args.surrogate_dir).glob("schnet_full_member_*.pt"))
            if members:
                from src.models.schnet_full import RealSchNetConfig, load_schnet_full

                device_spec = get_device(surrogate_device or loop_device_cfg or "cpu")
                surrogate_device_runtime = (
                    f"{device_spec.type}:{device_spec.index}" if device_spec.index is not None else device_spec.type
                )
                sch_cfg = RealSchNetConfig(device=surrogate_device_runtime)
                loaded = [
                    load_schnet_full(m, target_dim=len(loop_cfg.target_columns), cfg=sch_cfg) for m in members
                ]
                surrogate = SchNetSurrogateWrapper(loaded, surrogate_device_runtime, loop_cfg.target_columns)
                surrogate.is_schnet_full = True
            else:
                surrogate = SurrogateEnsemble.from_directory(Path(args.surrogate_dir), device=surrogate_device)

    generator = None
    generator3d = None
    generator3d_template = None
    fragment_vocab: Optional[Dict[str, int]] = None
    generator_device_runtime: Optional[str] = None
    if args.generator_ckpt and cfg.data.fragment_vocab:
        vocab_path = Path(cfg.data.fragment_vocab)
        if vocab_path.exists():
            with vocab_path.open("r", encoding="utf-8") as f:
                raw_vocab = json.load(f)
            if not isinstance(raw_vocab, dict) or not raw_vocab:
                raise ValueError(f"Fragment vocab at {vocab_path} is empty or not a mapping.")

            def _intlike(x: object) -> bool:
                try:
                    int(x)
                    return True
                except Exception:
                    return False

            sample_key, sample_val = next(iter(raw_vocab.items()))
            # Case A: frag -> idx mapping (values int-like)
            if _intlike(sample_val):
                fragment_vocab = {k: int(v) for k, v in raw_vocab.items()}
            # Case B: idx -> frag mapping (keys int-like, values strings)
            elif _intlike(sample_key):
                fragment_vocab = {v: int(k) for k, v in raw_vocab.items()}
                logging.getLogger(__name__).info(
                    "Reversed fragment vocab idx->frag mapping from %s.", vocab_path
                )
            else:
                raise ValueError(f"Unrecognised fragment_vocab format in {vocab_path}")
        if fragment_vocab:
            generator = _load_jtvae_from_ckpt(
                Path(args.generator_ckpt),
                len(fragment_vocab),
                cond_dim=len(loop_cfg.target_columns),
            )
            if generator_device:
                device_spec = get_device(generator_device)
                generator = generator.to(device_spec.target)
                generator_device_runtime = (
                    f"{device_spec.type}:{device_spec.index}" if device_spec.index is not None else device_spec.type
                )
            else:
                generator_device_runtime = None
    if getattr(args, "generator_3d_ckpt", None):
        from src.models.vae3d import VAE3D
        from src.data.featurization_3d_gen import build_gen3d_dataset
        data_cfg = cfg.data
        max_atoms = getattr(getattr(cfg, "generator3d", {}), "max_atoms", 100)
        gen3d_ckpt = Path(args.generator_3d_ckpt)
        generator3d = VAE3D(max_atoms=max_atoms)
        generator3d.load_state_dict(torch.load(gen3d_ckpt, map_location="cpu"))
        if generator_device:
            device_spec = get_device(generator_device)
            generator3d = generator3d.to(device_spec.target)
            if generator_device_runtime is None:
                generator_device_runtime = (
                    f"{device_spec.type}:{device_spec.index}" if device_spec.index is not None else device_spec.type
                )
        df_template = pd.read_csv(data_cfg.labelled)
        template_ds = build_gen3d_dataset(
            df_template,
            mol_col=getattr(data_cfg, "mol_column", "mol"),
            smiles_col=getattr(data_cfg, "smiles_column", getattr(data_cfg, "smile_column", "smile")),
            max_atoms=max_atoms,
        )
        if len(template_ds) > 0:
            import numpy as np

            gen3d_cfg = getattr(cfg, "generator3d", {}) or {}
            pool_size = int(getattr(gen3d_cfg, "template_pool_size", 256))
            seed = int(getattr(gen3d_cfg, "template_seed", 1337))
            z_pool = template_ds.zs
            mask_pool = template_ds.mask
            if pool_size > 0 and pool_size < len(template_ds):
                rng = np.random.default_rng(seed)
                idxs = rng.choice(len(template_ds), size=pool_size, replace=False)
                idxs = torch.as_tensor(idxs, dtype=torch.long)
                z_pool = z_pool.index_select(0, idxs)
                mask_pool = mask_pool.index_select(0, idxs)
            generator3d_template = {
                "z_pool": z_pool,
                "mask_pool": mask_pool,
            }
            logging.getLogger(__name__).info(
                "Loaded %d 3D templates for generator sampling.",
                int(z_pool.size(0)),
            )

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
                "allow_fallback": pipeline_data.get("allow_fallback", defaults.allow_fallback),
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
            dft_job_defaults = {
                "charge": pipeline_config.quantum.charge,
                "multiplicity": pipeline_config.quantum.multiplicity,
                "metadata": {
                    "engine": pipeline_config.quantum.engine,
                    "level_of_theory": pipeline_config.quantum.level_of_theory
                    or f"{pipeline_config.quantum.method}/{pipeline_config.quantum.basis}",
                },
            }

    loop = ActiveLearningLoop(
        surrogate=surrogate,
        labelled=labelled,
        pool=pool,
        config=loop_cfg,
        generator=generator,
        generator3d=generator3d,
        generator3d_template=generator3d_template,
        fragment_vocab=fragment_vocab,
        dft=dft,
        generator_device=generator_device_runtime,
        dft_job_defaults=dft_job_defaults,
    )
    loop.run(args.iterations)
    loop.save_history()
    if qc_manager is not None:
        qc_manager.shutdown()
    print("Active learning completed.")


def main() -> None:
    # Parse log level first so it is accepted regardless of position (before/after subcommands)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--log-level", default="INFO", help="Global log level (case-insensitive).")
    pre_args, remaining = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="OSC discovery toolkit", parents=[pre_parser])
    parser.set_defaults(log_level=pre_args.log_level)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train-surrogate")
    train_parser.add_argument("--config", default="configs/train_conf.yaml")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--device",
        default=None,
        help="Device for surrogate training (e.g. 'auto', 'cuda', 'cuda:0', 'cpu').",
    )
    train_parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable AMP mixed precision (default: taken from config).",
    )
    train_parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable torch.compile kernel fusion (default: from config).",
    )
    train_parser.add_argument(
        "--compile-mode",
        default=None,
        help="torch.compile mode (default inherits from config).",
    )
    train_parser.add_argument(
        "--compile-fullgraph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Request fullgraph compilation when using torch.compile.",
    )

    train3d_parser = subparsers.add_parser("train-surrogate-3d")
    train3d_parser.add_argument("--config", default="configs/train_conf_3d.yaml")
    train3d_parser.add_argument(
        "--device",
        default=None,
        help="Device for 3D surrogate training (e.g., 'auto', 'cuda', 'cpu'). Overrides config.",
    )

    train3d_full_parser = subparsers.add_parser("train-surrogate-3d-full")
    train3d_full_parser.add_argument("--config", default="configs/train_conf_3d_full.yaml")
    train3d_full_parser.add_argument(
        "--device",
        default=None,
        help="Device for full SchNet surrogate training (e.g., 'auto', 'cuda', 'cpu'). Overrides config.",
    )

    gen_parser = subparsers.add_parser("train-generator")
    gen_parser.add_argument("--config", default="configs/gen_conf.yaml")
    gen_parser.add_argument(
        "--device",
        default=None,
        help="Device for JT-VAE training (e.g. 'auto', 'cuda', 'cpu').",
    )
    gen_parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable AMP mixed precision for JT-VAE (default: config).",
    )
    gen_parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable torch.compile for JT-VAE (default: config).",
    )
    gen_parser.add_argument(
        "--compile-mode",
        default=None,
        help="torch.compile mode for JT-VAE (default: config).",
    )
    gen_parser.add_argument(
        "--compile-fullgraph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fullgraph toggle for torch.compile when training JT-VAE.",
    )

    gen3d_parser = subparsers.add_parser("train-generator-3d")
    gen3d_parser.add_argument("--config", default="configs/gen_conf_3d.yaml")
    gen3d_parser.add_argument(
        "--device",
        default=None,
        help="Device for 3D generator training (e.g., 'auto', 'cuda', 'cpu').",
    )

    al_parser = subparsers.add_parser("active-loop")
    al_parser.add_argument("--config", default="configs/active_learn.yaml")
    al_parser.add_argument("--surrogate-dir", default="models/surrogate")
    al_parser.add_argument("--generator-ckpt", default=None)
    al_parser.add_argument("--generator-3d-ckpt", default=None)
    al_parser.add_argument("--iterations", type=int, default=5)
    al_parser.add_argument("--use-pseudo-dft", action="store_true")
    al_parser.add_argument(
        "--device",
        default=None,
        help="Device override for both surrogate and generator inference.",
    )
    al_parser.add_argument(
        "--surrogate-device",
        default=None,
        help="Device override for surrogate inference (takes precedence over --device).",
    )
    al_parser.add_argument(
        "--generator-device",
        default=None,
        help="Device override for generator sampling (takes precedence over --device).",
    )

    args = parser.parse_args(remaining)
    log_level_name = str(args.log_level).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    setup_logging(level=log_level)

    if args.command == "train-surrogate":
        train_surrogate(args)
    elif args.command == "train-surrogate-3d":
        train_surrogate_3d(args)
    elif args.command == "train-surrogate-3d-full":
        train_surrogate_3d_full(args)
    elif args.command == "train-generator":
        train_generator(args)
    elif args.command == "train-generator-3d":
        train_generator_3d(args)
    elif args.command == "active-loop":
        run_active_loop(args)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
