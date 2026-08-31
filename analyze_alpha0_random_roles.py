"""
Independent analysis for rho=0, alpha=0, gamma=0.1.

Research question
-----------------
When alpha=0, role assignment is random (w does not affect who is proposer).
Yet final mean_p / mean_q still fall as deposit c rises. Using *existing*
timeseries/summary CSVs only, summarize how population-level fairness and
role-payoff statistics co-vary with c.

Data limits
-----------
Saved timeseries have population means only. Individual role-exposure counts
(prop_counts / resp_counts) were not exported, so this script cannot measure
per-agent exposure variance directly. It uses:
  mean_proposer_payoff, mean_responder_payoff, mean_R, mean_payoff, ...
as the available proxies for realized role-payoff structure feeding Fermi update.

Outputs (under mechanism_analysis_alpha0/):
  1) final_by_repeat.csv
  2) final_summary_by_c.csv          (mean/std/sem across repeats)
  3) final_mechanism_indicators.csv  (derived gaps + trend vs c)
  4) dynamics_mean_sem.csv           (repeat-averaged trajectories at selected gens)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_cpp"
OUT_DIR = ROOT / "mechanism_analysis_alpha0"

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

# Snapshot-like gens for dynamics (plus every 1000 thereafter up to T-1 if present)
BASE_GENS = [0, 9, 99, 999, 9999, 49999, 99999]

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


def read_last_line_row(path: Path) -> dict:
    with path.open("rb") as f:
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"{path.name}: empty")
        f.seek(0, 2)
        size = f.tell()
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
        raise ValueError(f"{path.name}: missing {missing}; got {header}")
    values = [v.strip() for v in last_line.split(",")]
    if len(values) != len(header):
        df = pd.read_csv(path, usecols=["gen", *METRICS])
        s = df.loc[df["gen"].idxmax()]
        return {"gen": int(s["gen"]), **{m: float(s[m]) for m in METRICS}}

    raw = {h: values[i] for i, h in enumerate(header)}
    return {"gen": int(float(raw["gen"])), **{m: float(raw[m]) for m in METRICS}}


def discover_folders() -> list[tuple[Path, float]]:
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(RESULTS_DIR)
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
        matched = next((c for c in TARGET_CS if nearly_equal(params["c"], c, C_TOL)), None)
        if matched is None:
            continue
        found.append((folder, matched))
    return found


def list_timeseries(folder: Path, c_canon: float) -> list[tuple[Path, int]]:
    out: list[tuple[Path, int]] = []
    for path in sorted(folder.glob("*_timeseries.csv")):
        meta = parse_timeseries_name(path.name)
        if meta is None:
            continue
        if not (
            nearly_equal(meta["rho"], TARGET_RHO)
            and nearly_equal(meta["alpha"], TARGET_ALPHA)
            and nearly_equal(meta["gamma"], TARGET_GAMMA)
            and nearly_equal(meta["c"], c_canon, C_TOL)
        ):
            continue
        out.append((path, meta["repeat"]))
    return out


def collect_final_by_repeat() -> pd.DataFrame:
    rows: list[dict] = []
    for folder, c in discover_folders():
        files = list_timeseries(folder, c)
        for path, repeat in files:
            last = read_last_line_row(path)
            row = {"c": c, "repeat": repeat, "gen": last["gen"], "source_file": path.name}
            row.update({m: last[m] for m in METRICS})
            # Derived population-level role-payoff gap (NOT the same as mean_R)
            row["role_payoff_gap"] = last["mean_proposer_payoff"] - last["mean_responder_payoff"]
            row["mean_R_minus_role_gap"] = last["mean_R"] - row["role_payoff_gap"]
            rows.append(row)
        print(f"[ok] {folder.name}: {len(files)} repeats")
    if not rows:
        raise RuntimeError("No matching alpha=0 timeseries found.")
    return pd.DataFrame(rows).sort_values(["c", "repeat"]).reset_index(drop=True)


def mean_std_sem(vals: np.ndarray) -> tuple[float, float, float]:
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n)
    return mean, std, sem


def summarize_by_c(final_df: pd.DataFrame) -> pd.DataFrame:
    extra = ["role_payoff_gap", "mean_R_minus_role_gap"]
    all_metrics = METRICS + extra
    rows = []
    for c in TARGET_CS:
        sub = final_df.loc[np.isclose(final_df["c"], c, atol=C_TOL)]
        n = len(sub)
        out: dict = {"c": c, "n_repeats": n}
        for m in all_metrics:
            mean, std, sem = mean_std_sem(sub[m].to_numpy(dtype=float) if n else np.array([]))
            out[f"{m}_mean"] = mean
            out[f"{m}_std"] = std
            out[f"{m}_sem"] = sem
        rows.append(out)

    cols = ["c", "n_repeats"]
    for m in all_metrics:
        cols.extend([f"{m}_mean", f"{m}_std", f"{m}_sem"])
    return pd.DataFrame(rows)[cols]


def mechanism_indicators(final_df: pd.DataFrame, summary: pd.DataFrame):
    """
    Compact table for the random-role / deposit mechanism story.
    Spearman vs c uses all repeat-level final points.
    """
    ind = summary[
        [
            "c",
            "n_repeats",
            "mean_p_mean",
            "mean_q_mean",
            "mean_w_mean",
            "mean_payoff_mean",
            "sample_success_rate_mean",
            "mean_proposer_payoff_mean",
            "mean_responder_payoff_mean",
            "role_payoff_gap_mean",
            "mean_R_mean",
            "mean_R_minus_role_gap_mean",
        ]
    ].copy()

    def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        try:
            from scipy import stats

            rho_s, p_s = stats.spearmanr(x, y)
            return float(rho_s), float(p_s)
        except Exception:
            rx = pd.Series(x).rank().to_numpy()
            ry = pd.Series(y).rank().to_numpy()
            if len(rx) < 2:
                return np.nan, np.nan
            return float(np.corrcoef(rx, ry)[0, 1]), np.nan

    trend_rows = []
    for m in METRICS + ["role_payoff_gap"]:
        rho_s, p_s = spearman(final_df["c"].to_numpy(), final_df[m].to_numpy())
        trend_rows.append({"metric": m, "spearman_vs_c": rho_s, "spearman_p": p_s})
    return ind, pd.DataFrame(trend_rows)


def selected_gens_from_file(path: Path, want: set[int]) -> pd.DataFrame:
    """Stream timeseries and keep only selected generation rows."""
    kept = []
    usecols = ["gen", *METRICS]
    # chunked read to limit memory
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=20000):
        sub = chunk.loc[chunk["gen"].isin(want)]
        if not sub.empty:
            kept.append(sub)
    if not kept:
        return pd.DataFrame(columns=usecols)
    return pd.concat(kept, ignore_index=True)


def collect_dynamics() -> pd.DataFrame:
    """Repeat-averaged (±sem) trajectories at selected generations only."""
    want = set(BASE_GENS)

    records = []
    for folder, c in discover_folders():
        files = list_timeseries(folder, c)
        per_repeat = []
        for path, repeat in files:
            df = selected_gens_from_file(path, want)
            if df.empty:
                continue
            df = df.copy()
            df["c"] = c
            df["repeat"] = repeat
            df["role_payoff_gap"] = df["mean_proposer_payoff"] - df["mean_responder_payoff"]
            per_repeat.append(df)
        if not per_repeat:
            continue
        all_rep = pd.concat(per_repeat, ignore_index=True)
        for gen, gdf in all_rep.groupby("gen"):
            row = {"c": c, "gen": int(gen), "n_repeats": int(gdf["repeat"].nunique())}
            for m in METRICS + ["role_payoff_gap"]:
                mean, std, sem = mean_std_sem(gdf[m].to_numpy(dtype=float))
                row[f"{m}_mean"] = mean
                row[f"{m}_sem"] = sem
            records.append(row)
        print(f"[dyn] {folder.name}: aggregated {len(files)} repeats")

    return pd.DataFrame(records).sort_values(["c", "gen"]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Part A: final generation by repeat / by c ===")
    final_df = collect_final_by_repeat()
    final_path = OUT_DIR / "final_by_repeat.csv"
    final_df.to_csv(final_path, index=False)

    summary = summarize_by_c(final_df)
    summary_path = OUT_DIR / "final_summary_by_c.csv"
    summary.to_csv(summary_path, index=False)

    print("=== Part B: mechanism indicator table + Spearman vs c ===")
    ind, trend = mechanism_indicators(final_df, summary)
    ind_path = OUT_DIR / "final_mechanism_indicators.csv"
    trend_path = OUT_DIR / "spearman_metric_vs_c.csv"
    ind.to_csv(ind_path, index=False)
    trend.to_csv(trend_path, index=False)

    print("=== Part C: dynamics (selected gens, mean±sem over repeats) ===")
    dyn = collect_dynamics()
    dyn_path = OUT_DIR / "dynamics_mean_sem.csv"
    dyn.to_csv(dyn_path, index=False)

    print("\nWrote:")
    for p in [final_path, summary_path, ind_path, trend_path, dyn_path]:
        print(f"  {p}")

    # Brief console view of the fairness drop under random roles
    view = summary[
        [
            "c",
            "n_repeats",
            "mean_p_mean",
            "mean_q_mean",
            "mean_proposer_payoff_mean",
            "mean_responder_payoff_mean",
            "role_payoff_gap_mean",
            "mean_R_mean",
            "mean_payoff_mean",
        ]
    ]
    print("\nFinal-state means (alpha=0, rho=0, gamma=0.1):")
    print(view.to_string(index=False))
    print("\nSpearman(metric, c) across all repeats:")
    print(trend.to_string(index=False))


if __name__ == "__main__":
    main()
