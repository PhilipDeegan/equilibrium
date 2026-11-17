"""
Long-running convergence study runner with incremental save and resume.

Usage
-----
    from equilibrium.physical.atom.convergence_runner import ConvergenceRunner, bohr_tasks

    tasks   = bohr_tasks(deltas=(0.3, 0.2, 0.15, 0.1), box_factor=3, n_levels=2)
    runner  = ConvergenceRunner("results/bohr", max_workers=4)
    results = runner.run(tasks)
    runner.print_table(results)

    # Optional live status endpoint:
    runner = ConvergenceRunner("results/bohr", max_workers=4, serve=True, port=8421)
"""

import json
import time
from multiprocessing import Pool
from pathlib import Path

# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------


def bohr_tasks(deltas=(0.3, 0.2, 0.15), box_factor: int = 3, n_levels: int = 2):
    """Return list of task dicts for the Bohr convergence study."""
    tasks = []
    for delta in deltas:
        for n in range(1, n_levels + 1):
            radius = float(max(6, box_factor * n**2))
            tasks.append({"delta": delta, "n": n, "radius": radius})
    return tasks


# ---------------------------------------------------------------------------
# Worker — must be module-level for multiprocessing pickling
# ---------------------------------------------------------------------------


def _worker(task):
    import numpy as np
    from scipy.sparse.linalg import eigsh
    from equilibrium.physical.atom.bohr_kelvin import build_hamiltonian

    delta = task["delta"]
    n = task["n"]
    radius = task["radius"]
    exact = -1.0 / (2 * n**2)

    t0 = time.monotonic()
    H, _ = build_hamiltonian(delta, radius)
    N = H.shape[0]
    n_eig = min(n + 10, N - 1)
    eigs = np.sort(eigsh(H, k=n_eig, which="SA", return_eigenvectors=False))
    energy = float(eigs[np.argmin(np.abs(eigs - exact))])
    elapsed = time.monotonic() - t0

    return {
        "delta": delta,
        "n": n,
        "radius": radius,
        "N": N,
        "energy": energy,
        "exact": exact,
        "error_pct": 100.0 * abs(energy - exact) / abs(exact),
        "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ConvergenceRunner:
    def __init__(
        self,
        results_dir: str,
        max_workers: int = 4,
        serve: bool = False,
        port: int = 8421,
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self._status_path = self.results_dir / "status.json"

        if serve:
            self._start_server(port)

    # ------------------------------------------------------------------
    # Paths and persistence
    # ------------------------------------------------------------------

    def _result_path(self, delta, n):
        return self.results_dir / f"bohr_d{delta:.4f}_n{n}.json"

    def _is_done(self, task):
        return self._result_path(task["delta"], task["n"]).exists()

    def _save(self, result):
        path = self._result_path(result["delta"], result["n"])
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

    def load_all(self):
        results = []
        for p in sorted(self.results_dir.glob("bohr_d*.json")):
            with open(p) as f:
                results.append(json.load(f))
        return sorted(results, key=lambda r: (r["delta"], r["n"]))

    # ------------------------------------------------------------------
    # Status file
    # ------------------------------------------------------------------

    def _write_status(self, total, done, running, pending):
        status = {
            "total": total,
            "done": done,
            "running": running,
            "pending": pending,
            "updated": time.strftime("%H:%M:%S"),
        }
        with open(self._status_path, "w") as f:
            json.dump(status, f, indent=2)

    # ------------------------------------------------------------------
    # Optional FastAPI server in a daemon thread
    # ------------------------------------------------------------------

    def _start_server(self, port):
        try:
            import threading
            import uvicorn
            from fastapi import FastAPI
            from fastapi.responses import JSONResponse

            app = FastAPI()

            @app.get("/status")
            def status():
                if self._status_path.exists():
                    return JSONResponse(json.loads(self._status_path.read_text()))
                return JSONResponse({"status": "not started"})

            @app.get("/results")
            def results():
                return JSONResponse(self.load_all())

            cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
            t = threading.Thread(target=uvicorn.Server(cfg).run, daemon=True)
            t.start()
            print(f"Status endpoint: http://127.0.0.1:{port}/status", flush=True)
        except ImportError:
            print(
                "fastapi/uvicorn not installed — status endpoint disabled", flush=True
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, tasks):
        pending = [t for t in tasks if not self._is_done(t)]
        n_done = len(tasks) - len(pending)
        n_total = len(tasks)

        print(
            f"Tasks: {n_total} total  |  {n_done} already done  |  {len(pending)} pending",
            flush=True,
        )
        self._write_status(n_total, n_done, 0, len(pending))

        if not pending:
            return self.load_all()

        with Pool(self.max_workers) as pool:
            for result in pool.imap_unordered(_worker, pending):
                self._save(result)
                n_done += 1
                remaining = len(pending) - n_done
                print(
                    f"  [done {n_done}/{n_total}]  "
                    f"delta={result['delta']}  n={result['n']}  "
                    f"N={result['N']:,}  "
                    f"E={result['energy']:.6f}  err={result['error_pct']:.2f}%  "
                    f"({result['elapsed_s']}s)",
                    flush=True,
                )
                self._write_status(n_total, n_done, 0, remaining)

        return self.load_all()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_table(self, results):
        from equilibrium.physical.atom.bohr_kelvin import planck_extrapolation

        HAR_TO_NM = 1240.0 / 27.2114

        by_delta = {}
        for r in results:
            by_delta.setdefault(r["delta"], {})[r["n"]] = r

        n_levels = max(r["n"] for r in results)
        print(f"\n{'delta':>8}  {'N(n=1)':>9}", end="")
        for n in range(1, n_levels + 1):
            print(f"  E{n}(exact={-1/(2*n**2):+.4f})", end="")
        if n_levels >= 2:
            print("  λ_Lα(nm)", end="")
        print()

        deltas, errs1 = [], []
        for delta in sorted(by_delta):
            lvls = by_delta[delta]
            if not all(n in lvls for n in range(1, n_levels + 1)):
                continue
            print(f"{delta:>8.3f}  {lvls[1]['N']:>9,}", end="")
            for n in range(1, n_levels + 1):
                e, ep = lvls[n]["energy"], lvls[n]["error_pct"]
                print(f"  {e:+.6f}({ep:.1f}%)", end="")
            if n_levels >= 2:
                lam = HAR_TO_NM / (lvls[2]["energy"] - lvls[1]["energy"])
                print(f"  {lam:>9.3f}", end="")
            print()
            deltas.append(delta)
            errs1.append(lvls[1]["error_pct"])

        if len(deltas) >= 2:
            _, ex1 = planck_extrapolation(deltas, errs1)
            print(f"\nPlanck extrapolation n=1: {ex1:.2e}%")
