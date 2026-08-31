"""
Mechanism analysis — Part 1: final-state summary across repeats for each deposit c.

Reads existing simulation outputs only (no C++ / no re-run / no writes to results_cpp).
Target experiment: rho=0, alpha=6, gamma=0.1, c in {0, 0.05, ..., 0.40}.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# --- paths ---
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_cpp"
OUT_DIR = ROOT / "mechanism_analysis"

# --- target experiment ---
TARGET_RHO = 0.0
TARGET_ALPHA = 0.0
TARGET_GAMMA = 0.1
TARGET_CS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
C_TOL = 1e-6
PARAM_TOL = 1e-6

METRICS = [
    "mean_p",
    "mean_q",
    "mean_w",
    "mean_payoff",
    "sample_success_rate",
    "mean_proposer_payoff",
    "mean_responder_payoff",
    "mean_R",
]

# Actual on-disk names (checked):
#   folder: L100_T100000_c0.00_rho0.00_gamma0.10_alpha6.00
#   timeseries: c=0.000000_gamma=0.100000_rho=0.000000_alpha=6.000000_repeat_0_timeseries.csv
#   summary: summary_c=0.000000_gamma=0.100000_rho=0.000000_alpha=6.000000.csv
FOLDER_RE = re.compile(
    r"c(?P<c>[0-9.]+)_rho(?P<rho>[0-9.]+)_gamma(?P<gamma>[0-9.]+)_alpha(?P<alpha>[0-9.]+)",
    re.IGNORECASE,
)
TIMESERIES_RE = re.compile(
    r"c=(?P<c>[0-9.]+)_gamma=(?P<gamma>[0-9.]+)_rho=(?P<rho>[0-9.]+)_alpha=(?P<alpha>[0-9.]+)"
    r"_repeat_(?P<repeat>\d+)_timeseries\.csv$",
    re.IGNORECASE,
)


def nearly_equal(a: float, b: float, tol: float = PARAM_TOL) -> bool:
    return abs(a - b) <= tol


def parse_folder_params(name: str) -> dict[str, float] | None:
    m = FOLDER_RE.search(name)
    if not m:
        return None
    return {k: float(m.group(k)) for k in ("c", "rho", "gamma", "alpha")}


def parse_timeseries_name(name: str) -> dict | None:
    m = TIMESERIES_RE.search(name)
    if not m:
        return None
    return {
        "c": float(m.group("c")),
        "rho": float(m.group("rho")),
        "gamma": float(m.group("gamma")),
        "alpha": float(m.group("alpha")),
        "repeat": int(m.group("repeat")),
    }


def read_timeseries_last_generation(path: Path) -> pd.Series:
    """
    Extract the final generation row (max gen).

    Files are large (~1e5 rows). We read the header, then only the last data
    line (writers append in increasing gen order). If gen is not the max of a
    small tail check, fall back to a full scan of needed columns.
    """
    with path.open("rb") as f:
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"{path.name}: empty file")
        f.seek(0, 2)
        size = f.tell()
        # walk back to previous newline
        pos = max(0, size - 2)
        while pos > 0:
            f.seek(pos)
            if f.read(1) == b"\n":
                break
            pos -= 1
        last_line = f.read().decode("utf-8", errors="replace").strip()

    header = [h.strip() for h in header_line.decode("utf-8", errors="replace").strip().split(",")]
    missing = [c for c in (["gen", *METRICS]) if c not in header]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; got {header}")
    if not last_line:
        raise ValueError(f"{path.name}: no data rows")

    values = [v.strip() for v in last_line.split(",")]
    if len(values) != len(header):
        # rare: fall back to pandas full read of needed cols
        df = pd.read_csv(path, usecols=["gen", *METRICS])
        return df.loc[df["gen"].idxmax()]

    raw = {h: values[i] for i, h in enumerate(header)}
    out = {"gen": int(float(raw["gen"]))}
    for m in METRICS:
        out[m] = float(raw[m])
    return pd.Series(out)


def discover_target_folders() -> list[tuple[Path, float]]:
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")

    found: list[tuple[Path, float]] = []
    for folder in sorted(RESULTS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        params = parse_folder_params(folder.name)
        if params is None:
            continue
        if not (
            nearly_equal(params["rho"], TARGET_RHO)
            and nearly_equal(params["alpha"], TARGET_ALPHA)
            and nearly_equal(params["gamma"], TARGET_GAMMA)
        ):
            continue
        # Keep only requested deposit sizes
        matched_c = None
        for c in TARGET_CS:
            if nearly_equal(params["c"], c, C_TOL):
                matched_c = c
                break
        if matched_c is None:
            continue
        found.append((folder, matched_c))
    return found


def collect_final_rows() -> pd.DataFrame:
    folders = discover_target_folders()
    if not folders:
        raise RuntimeError(
            f"No matching folders under {RESULTS_DIR} "
            f"(need rho={TARGET_RHO}, alpha={TARGET_ALPHA}, gamma={TARGET_GAMMA})"
        )

    rows: list[dict] = []
    for folder, c_canon in folders:
        ts_files = sorted(folder.glob("*_timeseries.csv"))
        if not ts_files:
            # fallback pattern
            ts_files = sorted(folder.glob("*timeseries*.csv"))

        n_used = 0
        for path in ts_files:
            meta = parse_timeseries_name(path.name)
            if meta is None:
                print(f"[skip] unrecognized timeseries name: {path.name}")
                continue
            if not (
                nearly_equal(meta["rho"], TARGET_RHO)
                and nearly_equal(meta["alpha"], TARGET_ALPHA)
                and nearly_equal(meta["gamma"], TARGET_GAMMA)
                and nearly_equal(meta["c"], c_canon, C_TOL)
            ):
                print(f"[skip] param mismatch in file: {path.name}")
                continue

            last = read_timeseries_last_generation(path)
            row = {
                "c": c_canon,
                "repeat": meta["repeat"],
                "gen": int(last["gen"]),
                "source_file": path.name,
            }
            for m in METRICS:
                row[m] = float(last[m])
            rows.append(row)
            n_used += 1

        print(f"[ok] {folder.name}: used {n_used} timeseries files (c={c_canon})")

    if not rows:
        raise RuntimeError("No timeseries final rows collected.")
    return pd.DataFrame(rows).sort_values(["c", "repeat"]).reset_index(drop=True)


def summarize_by_c(final_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict] = []
    for c in TARGET_CS:
        sub = final_df.loc[np.isclose(final_df["c"], c, atol=C_TOL)]
        n = len(sub)
        out: dict = {"c": c, "n_repeats": n}
        if n == 0:
            for m in METRICS:
                out[f"{m}_mean"] = np.nan
                out[f"{m}_std"] = np.nan
                out[f"{m}_sem"] = np.nan
            summary_rows.append(out)
            continue

        for m in METRICS:
            vals = sub[m].to_numpy(dtype=float)
            # ddof=1: sample std across repeats
            std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            mean = float(np.mean(vals))
            sem = std / np.sqrt(n) if n > 0 else np.nan
            out[f"{m}_mean"] = mean
            out[f"{m}_std"] = std
            out[f"{m}_sem"] = sem
        summary_rows.append(out)

    col_order = ["c", "n_repeats"]
    for m in METRICS:
        col_order.extend([f"{m}_mean", f"{m}_std", f"{m}_sem"])
    return pd.DataFrame(summary_rows)[col_order]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    final_df = collect_final_rows()
    # Optional audit table (repeat-level finals); does not modify originals
    audit_path = OUT_DIR / "mechanism_final_by_repeat_alpha0.csv"
    final_df.to_csv(audit_path, index=False)

    summary = summarize_by_c(final_df)
    out_path = OUT_DIR / "mechanism_final_summary_alpha0.csv"
    summary.to_csv(out_path, index=False)

    print(f"\nWrote: {out_path}")
    print(f"Wrote: {audit_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
