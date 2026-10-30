"""Paths, seeds and the pink/purple palette shared by the whole project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "CS_data.txt"
VARIABLE_LIST = ROOT / "data" / "variable_list.csv"
OUT_DIR = ROOT / "outputs"
TEMPLATE = ROOT / "src" / "templates" / "dashboard_fragment.html"

SEED = 42
TEST_SIZE = 0.25

PALETTE = {
    "blush": "#fbe3f0",
    "pink": "#f06ab0",
    "orchid": "#c27be0",
    "violet": "#8e4ec6",
    "plum": "#3b1f5c",
}
