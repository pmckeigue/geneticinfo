"""
test_blas_threads.py — standalone diagnostic for BLAS multi-threading in forked workers.

Run this script (or paste it into a Jupyter cell) to verify that BLAS thread
count can be set correctly in the forked subprocesses that sample_posterior
uses, and that it actually produces a speedup.

Usage:
    python test_blas_threads.py
"""

import multiprocessing as mp
import os
import sys
import time
import numpy as np

# ── worker ────────────────────────────────────────────────────────────────────

def _blas_test_worker(n_threads):
    """
    Run inside a forked subprocess.  Sets BLAS threads, reports the actual
    count via threadpoolctl, and times a 1500×1500 Cholesky factorisation.
    """
    import ctypes, sys, os

    # --- Try each mechanism in turn ---

    # 1. env vars
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)

    # 2. MKL direct load (by name then by /proc/self/maps path)
    def _try_mkl(lib):
        try:
            lib.MKL_Set_Num_Threads(ctypes.c_int(n_threads))
            return True
        except Exception:
            return False

    mkl_ok = False
    for mkl_name in ("libmkl_rt.so", "libmkl_rt.so.2", "libmkl_rt.so.1", "mkl_rt"):
        try:
            if _try_mkl(ctypes.CDLL(mkl_name)):
                mkl_ok = True
                break
        except Exception:
            pass

    if not mkl_ok:
        import re
        try:
            with open("/proc/self/maps") as f:
                for line in f:
                    m = re.search(r'(/[^\s]*libmkl_rt[^\s]*\.so[0-9.]*)', line)
                    if m:
                        try:
                            if _try_mkl(ctypes.CDLL(m.group(1))):
                                mkl_ok = True
                                break
                        except Exception:
                            pass
        except Exception:
            pass

    # 3a. OpenBLAS via global symbol table
    openblas_ok = False
    try:
        ctypes.CDLL(None).openblas_set_num_threads(ctypes.c_int(n_threads))
        openblas_ok = True
    except Exception:
        pass

    # 3b. mkl-service
    mkl_service_ok = False
    try:
        import mkl as _mkl
        _mkl.set_num_threads(n_threads)
        mkl_service_ok = True
    except Exception:
        pass

    # 4. threadpoolctl
    threadpoolctl_ok = False
    sys.unraisablehook = lambda _: None
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n_threads, user_api="blas")
        threadpoolctl_ok = True
    except Exception:
        pass

    # Report what worked
    methods = {
        "mkl_direct": mkl_ok,
        "openblas_cdll_none": openblas_ok,
        "mkl_service": mkl_service_ok,
        "threadpoolctl": threadpoolctl_ok,
    }

    # Actual thread count via threadpoolctl
    try:
        from threadpoolctl import threadpool_info
        info = [x for x in threadpool_info() if x.get("user_api") == "blas"]
        actual = [(x["internal_api"], x["num_threads"]) for x in info]
    except Exception:
        actual = "unavailable"

    # Time a 1500×1500 Cholesky
    A = np.random.default_rng(0).standard_normal((1500, 1500))
    M = A @ A.T + 1500 * np.eye(1500)
    _ = np.linalg.cholesky(M)  # warm-up
    t0 = time.perf_counter()
    for _ in range(10):
        np.linalg.cholesky(M)
    ms = (time.perf_counter() - t0) / 10 * 1000

    print(
        f"  n_threads={n_threads:2d} | actual={actual} | "
        f"chol(1500)={ms:6.1f} ms | "
        f"methods={methods}",
        flush=True,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Python: {sys.version}")
    print(f"numpy:  {np.__version__}")
    print(f"CPUs:   {os.cpu_count()}")

    try:
        from threadpoolctl import threadpool_info
        sys.unraisablehook = lambda _: None
        info = threadpool_info()
        print(f"Parent BLAS: {[(x['internal_api'], x.get('filepath','?'), x['num_threads']) for x in info if x.get('user_api')=='blas']}")
    except Exception as e:
        print(f"threadpool_info failed in parent: {e}")

    print()
    print("Testing BLAS thread count in forked workers:")

    ctx = mp.get_context("fork")
    for n in [1, 2, 4, 8]:
        with ctx.Pool(1) as pool:
            pool.apply(_blas_test_worker, args=(n,))

    print()
    print("If chol time decreases as n_threads increases, multi-threading works.")
    print("If it stays flat, check 'actual' — if it stays at 1, the thread-setting")
    print("mechanism failed for your BLAS implementation.")


if __name__ == "__main__":
    main()
