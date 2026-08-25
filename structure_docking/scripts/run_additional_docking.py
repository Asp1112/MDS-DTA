from pathlib import Path

import run_corrected_docking as docking


ROOT = Path(r"structure_docking/11_additional_6")
docking.ANALYSIS = ROOT / "06_structure_analysis" / "additional_structure_analysis.json"
docking.OUT = ROOT / "07_docking"
docking.REUSE_RANKS = set()

if __name__ == "__main__":
    docking.main()
