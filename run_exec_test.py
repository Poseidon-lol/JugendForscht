import tempfile
from pathlib import Path
from src.qc.config import GeometryConfig, QuantumTaskConfig
from src.qc.executors import OrcaExecutor
from src.qc.geometry import generate_3d_geometry
import yaml

cfg = yaml.safe_load(Path('configs/qc_pipeline.yaml').read_text())
geom_cfg = GeometryConfig(**cfg.get('geometry', {}))
quant_cfg = QuantumTaskConfig(**cfg.get('quantum', {}))
geometry = generate_3d_geometry('c1ccccc1', geom_cfg)
executor = OrcaExecutor()
executor.executable = str(quant_cfg.executable)
with tempfile.TemporaryDirectory(prefix='orca_exec_test_') as tmp:
    tmp_path = Path(tmp)
    input_path = tmp_path/'input.inp'
    output_path = tmp_path/'output.out'
    executor._write_input(input_path, geometry, quant_cfg)
    print('running in', tmp_path)
    try:
        executor._execute(quant_cfg, input_path, output_path, tmp_path)
    except Exception as exc:
        print('exception:', exc)
        if output_path.exists():
            print('output contents:\n', output_path.read_text())
        raise
    else:
        print('success, output length:', len(output_path.read_text()))
