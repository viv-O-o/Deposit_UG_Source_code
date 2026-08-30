"""
Batch-run Evo_Ug_Sim.exe from params.csv using multiprocessing.

Does not modify the C++ model. Each CSV row launches one exe process and
feeds parameters through stdin (same order as the interactive prompts).

Required CSV columns:
    L, T, c, rho, gamma, alpha, repeats
"""

import os
import subprocess
import multiprocessing
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# Point this to your built Release exe
EXE_PATH = Path(r"D:\VisualStudio\work_place\Evo_Ug_Sim\Evo_Ug_Sim\x64\Release\Evo_Ug_Sim.exe")

# Directory where results_cpp/ will be created (exe uses relative paths)
WORK_DIR = Path(r"D:\VisualStudio\work_place\Evo_Ug_Sim\Evo_Ug_Sim")

PARAMS_CSV = WORK_DIR / "params.csv"
NUM_CORES = 12


def run_simulation(args):
    """
    args: (L, T, c, rho, gamma, alpha, repeats)
    """
    L, T, c, rho, gamma, alpha, repeats = args

    # Same order as main.cpp cin prompts
    stdin_text = (
        f"{int(L)}\n"
        f"{int(T)}\n"
        f"{c}\n"
        f"{rho}\n"
        f"{gamma}\n"
        f"{alpha}\n"
        f"{int(repeats)}\n"
    )

    try:
        subprocess.run(
            [str(EXE_PATH)],
            input=stdin_text,
            text=True,
            check=True,
            cwd=str(WORK_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        err = e.stderr if isinstance(e.stderr, str) else ""
        print(f"Error running {args}: {e}\n{err}")
    except Exception as e:
        print(f"Error running {args}: {e}")


if __name__ == "__main__":
    if not EXE_PATH.exists():
        print(f"Error: exe not found: {EXE_PATH}")
        print("Build Release|x64 first, or update EXE_PATH in run_batch.py")
        raise SystemExit(1)

    if not PARAMS_CSV.exists():
        print(f"Error: {PARAMS_CSV} not found!")
        raise SystemExit(1)

    df = pd.read_csv(PARAMS_CSV)
    required = ["L", "T", "c", "rho", "gamma", "alpha", "repeats"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Error: params.csv missing columns: {missing}")
        raise SystemExit(1)

    tasks = [
        (
            row["L"],
            row["T"],
            row["c"],
            row["rho"],
            row["gamma"],
            row["alpha"],
            row["repeats"],
        )
        for _, row in df.iterrows()
    ]

    num_cores = min(NUM_CORES, max(1, len(tasks)))
    print(f"Starting {len(tasks)} tasks on {num_cores} cores...")
    print(f"Exe: {EXE_PATH}")
    print(f"Work dir (results_cpp): {WORK_DIR}")

    # Windows: guard is required for multiprocessing spawn
    with multiprocessing.Pool(processes=num_cores) as pool:
        list(tqdm(pool.imap(run_simulation, tasks), total=len(tasks)))

    print("Successfully completed all simulations.")
