#!/usr/bin/env python3
"""
download_swec.py — fetch seizure + interictal hours for the SWEC-ETHZ cohort.

Reads each patient's info file, derives which hourly files to pull, and downloads them.
Re-runnable: existing files are skipped (wget -c).
"""
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import scipy.io as sio

DATA_DIR = Path("datasets/swec-ethz")
PLAN_PATH = Path("scripts/download_manifest.json")
AVAILABLE_HOURS_PATH = DATA_DIR / "longterm-files.txt"
BASE_URL = "http://ieeg-swez.ethz.ch/long-term_dataset"
COHORT = ["ID03", "ID04", "ID05", "ID06", "ID07",
          "ID10", "ID12", "ID13", "ID16", "ID18"]
N_INTERICTAL = 4         # target interictal hours per patient
N_INTERICTAL_MIN = 3     # warn if fewer than this are pickable
INTERICTAL_BUFFER = 3    # hours of distance required from any seizure


def load_info(path: Path) -> dict:
    """Load a .mat file, falling back to mat73 for v7.3 files."""
    try:
        return sio.loadmat(str(path), simplify_cells=True)
    except NotImplementedError as e:
        raise RuntimeError(f"{path} is v7.3; install mat73 (`uv add mat73`)") from e


def seizure_hours(info: dict) -> set[int]:
    """
    Extract integer hour values for every seizure in the info file.

    SWEC info files store seizure_begin/seizure_end in seconds, so we
    floor-divide by 3600 to map to the hourly file index.
    """
    hours: set[int] = set()
    for key, val in info.items():
        if key.startswith("__"):
            continue
        if any(c in key.lower() for c in ["seizure", "szr", "onset", "offset"]):
            arr = val if hasattr(val, "__iter__") else [val]
            for v in arr:
                try:
                    hours.add(int(float(v) // 3600) + 1)  # +1 to shift from 0-indexed hours to 1-indexed files
                except (TypeError, ValueError):
                    continue
    return hours


def load_available_hours() -> dict[str, set[int]]:
    """Parse longterm-files.txt → {patient: {hour, ...}} of files served upstream."""
    pattern = re.compile(r"/(ID\d+)/\1_(\d+)h\.mat$")
    available: dict[str, set[int]] = {}
    for line in AVAILABLE_HOURS_PATH.read_text().splitlines():
        m = pattern.search(line.strip())
        if m:
            available.setdefault(m.group(1), set()).add(int(m.group(2)))
    return available


def pick_interictal(seizure_hrs: set[int], available_hrs: set[int],
                    n: int, buffer: int) -> list[int]:
    """Pick n hours spread across the recording, each ≥ buffer away from any seizure."""
    eligible = sorted(
        h for h in available_hrs
        if all(abs(h - s) >= buffer for s in seizure_hrs)
    )
    if not eligible:
        return []
    if len(eligible) <= n:
        return eligible
    step = len(eligible) / n
    return [eligible[int(i * step)] for i in range(n)]


def download(patient: str, hour: int) -> None:
    url = f"{BASE_URL}/{patient}/{patient}_{hour}h.mat"
    subprocess.run(["wget", "-c", "-P", str(DATA_DIR), url], check=False)


def remote_size(url: str) -> int | None:
    """HEAD request → Content-Length, or None if unreachable."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as r:
            return int(r.headers["Content-Length"])
    except Exception:
        return None


def verify(plan: dict[str, list[int]]) -> None:
    """Check plan against disk: missing files, size mismatches, unplanned extras."""
    pat = re.compile(r"^(ID\d+)_(\d+)h\.mat$")
    planned = {(p, h) for p, hours in plan.items() for h in hours}
    on_disk: dict[tuple[str, int], tuple[Path, int]] = {}
    for f in DATA_DIR.iterdir():
        m = pat.match(f.name)
        if m:
            on_disk[(m.group(1), int(m.group(2)))] = (f, f.stat().st_size)

    missing, mismatched, extras = [], [], []
    for p, h in sorted(planned):
        if (p, h) not in on_disk:
            missing.append(f"{p}_{h}h.mat")
            continue
        local = on_disk[(p, h)][1]
        remote = remote_size(f"{BASE_URL}/{p}/{p}_{h}h.mat")
        if remote is not None and local != remote:
            mismatched.append(f"{p}_{h}h.mat  local={local/1e6:.1f}MB  server={remote/1e6:.1f}MB")
    for key, (f, _) in sorted(on_disk.items()):
        if key not in planned:
            extras.append(f.name)

    print("\n=== verify ===")
    print(f"missing:    {len(missing)}")
    for x in missing:    
        print(f"  {x}")
    print(f"mismatched: {len(mismatched)}")
    for x in mismatched: 
        print(f"  {x}")
    print(f"unplanned:  {len(extras)}")
    for x in extras:     
        print(f"  {x}")
    if not (missing or mismatched or extras):
        print("OK — all planned files present at expected size, no extras.")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    available_hours = load_available_hours()
    plan: dict[str, list[int]] = {}

    for patient in COHORT:
        info_path = DATA_DIR / f"{patient}_info.mat"
        if not info_path.exists():
            print(f"Missing {info_path}; skipping {patient}")
            continue

        avail = available_hours.get(patient, set())
        if not avail:
            print(f"No upstream files listed for {patient}; skipping")
            continue

        info = load_info(info_path)
        sz = seizure_hours(info) & avail
        if not sz:
            print(f"No seizure hours parsed from {info_path}; check schema")
            continue

        interictal = pick_interictal(sz, avail, N_INTERICTAL, INTERICTAL_BUFFER)
        plan[patient] = sorted(sz | set(interictal))

        warn = " [WARN: low interictal]" if len(interictal) < N_INTERICTAL_MIN else ""
        print(f"{patient}: {len(sz)} seizure hr(s), {len(interictal)} interictal "
              f"→ {len(plan[patient])} files{warn}")

    total = sum(len(v) for v in plan.values())
    print(f"\nTotal: {total} files (~{total * 2} GB)\n")

    confirm = input("Proceed with download? [y/N] ")
    if confirm.lower() != "y":
        return

    PLAN_PATH.write_text(json.dumps(plan, indent=2))

    for i, (patient, hours) in enumerate(plan.items(), 1):
        print(f"\n[{i}/{len(plan)}] {patient}")
        for hour in hours:
            download(patient, hour)

    verify(plan)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        if not PLAN_PATH.exists():
            sys.exit(f"No plan at {PLAN_PATH}. Run the downloader first.")
        verify(json.loads(PLAN_PATH.read_text()))
    else:
        main()