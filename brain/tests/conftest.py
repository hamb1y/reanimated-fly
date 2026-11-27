import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BRAIN_SPEED_TARGET", "0")  # tests: run flat out
DATA = Path(os.environ.get("DATA_DIR", "/data"))
HAVE_DATA = (DATA / "2025_Connectivity_783.parquet").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="connectome data not mounted (set DATA_DIR)")
