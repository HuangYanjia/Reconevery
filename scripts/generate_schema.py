from pathlib import Path

from recon2sim.ir import SceneIR
from recon2sim.storage import atomic_write_json

atomic_write_json(Path("schemas/scene_ir.schema.json"), SceneIR.model_json_schema())
