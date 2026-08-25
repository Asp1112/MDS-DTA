"""Post-install numpy-2 compatibility patch for fairseq 0.10.2."""

import pathlib
import re


BASE = pathlib.Path(
    "/root/mds/.venv_dta/lib/python3.9/site-packages/fairseq")


def patch(name, patterns):
    path = BASE / name
    text = path.read_text()
    for pattern, new in patterns:
        text, n = re.subn(pattern, new, text, flags=re.MULTILINE)
        print(f"{name}: replaced {n} x {pattern!r}")
    path.write_text(text)
    print("patched:", name)


patch("data/indexed_dataset.py", [
    (r"^(\s*6: )np\.float,", r"\1np.float64,"),
    (r"^(\s*)np\.float: (np\.float64: 8|4),", r"\1np.float64: 8,"),
])
patch("data/data_utils.py", [
    (r"np\.int,", "np.int64,"),
])
patch("modules/dynamic_crf_layer.py", [
    (r'np\.float\("inf"\)', 'float("inf")'),
])
