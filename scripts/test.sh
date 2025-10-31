python - <<'PY'
from src.qc.config import PipelineConfig
from src.qc.pipeline import QCPipeline
from src.data.dft_int import DFTJobSpec

pipeline = QCPipeline(PipelineConfig())          # loads configs/qc_pipeline.yaml
job = DFTJobSpec(smiles="c1ccccc1", properties=["HOMO","LUMO","gap","IE","EA"])
result = pipeline.run(job)
print("status:", result.status)
print("props:", result.properties)
print("walltime:", result.wall_time)
print("metadata keys:", result.metadata.keys())
PY