from pathlib import Path
import sys
import yaml

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qc.config import GeometryConfig, QuantumTaskConfig, PipelineConfig
from src.qc.pipeline import QCPipeline
from src.data.dft_int import DFTJobSpec


def main() -> None:
    cfg = yaml.safe_load(Path("configs/qc_pipeline.yaml").read_text())
    geom = GeometryConfig(**cfg.get("geometry", {}))
    quantum = QuantumTaskConfig(**cfg.get("quantum", {}))
    pipe = PipelineConfig(
        geometry=geom,
        quantum=quantum,
        work_dir=Path(cfg.get("work_dir", "qc_runs")),
        max_workers=1,
        poll_interval=2.0,
        cleanup_workdir=False,
        store_metadata=True,
        tracked_properties=tuple(quantum.properties),
    )
    pipeline = QCPipeline(pipe)

    # Simple smoke test: water
    job = DFTJobSpec(smiles="O", properties=quantum.properties)
    res = pipeline.run(job)
    print("QC status:", res.status)
    print("Properties:", res.properties)
    print("Work dir:", res.metadata.get("workdir", "see engine output"))


if __name__ == "__main__":
    main()
