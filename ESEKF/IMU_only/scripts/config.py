import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # imu_only/

RAW_imu_RECS_DIR = PROJECT_ROOT.parent / "csv_files_of_neon_recordings"
OUTPUT_DIR = PROJECT_ROOT / "out"
FILTER_OUTPUT_FILE = OUTPUT_DIR / "plottable_filter_output.csv"

FILTER_FUNCS_FILE = OUTPUT_DIR / "filter_funcs_imu_only.pkl"

# make `import modules` work everywhere that imports this config
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
