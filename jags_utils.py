"""
Minimal Python interface to JAGS via the command-line executable.

Writes data in R dump format, runs JAGS as a subprocess, and parses
the CODA output files back into numpy arrays.  No pyjags dependency.

Usage
-----
    from jags_utils import run_jags

    samples = run_jags(
        model_file = "model_mix2.jags",
        data       = {"M": 42, "y": np.array([...]), ...},
        inits      = [{"s": 1.0, ...}, ...],   # one dict per chain
        n_adapt    = 1000,
        n_burnin   = 2000,
        n_samples  = 2000,
        monitors   = ["mu", "s", "beta0"],
        n_chains   = 4,
        workdir    = "/tmp/jags_run",
    )
    # samples["mu"] has shape (n_samples, n_chains)
"""

import os
import re
import subprocess
import tempfile
import shutil
import textwrap
import numpy as np


# ------------------------------------------------------------------ #
# R dump format writer
# ------------------------------------------------------------------ #
def _to_r_value(v):
    """Format a Python/numpy value as R dump text."""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        # Avoid -Inf / Inf / NaN in R dump
        return f"{float(v):.17g}"
    if isinstance(v, np.ndarray):
        flat = v.flatten(order="F")          # column-major (R default)
        vals = ", ".join(f"{x:.17g}" for x in flat)
        shape = ", ".join(str(d) for d in v.shape)
        if v.ndim == 1:
            return f"c({vals})"
        return f"structure(c({vals}), .Dim = c({shape}))"
    raise TypeError(f"Cannot serialise {type(v)}")


def write_r_dump(path, data_dict):
    """Write a dict of Python/numpy values to an R dump file."""
    lines = []
    for k, v in data_dict.items():
        lines.append(f'"{k}" <-\n{_to_r_value(v)}\n')
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ------------------------------------------------------------------ #
# CODA output parser
# ------------------------------------------------------------------ #
def _parse_coda(workdir, monitors, n_chains, n_samples):
    """
    Read JAGS CODA output files and return a dict of arrays.

    Files produced by 'coda *':
        CODAchain1.txt, CODAchain2.txt, ... (iteration + value columns)
        CODAindex.txt  (variable name + first/last row)

    Returns
    -------
    dict  { varname : ndarray shape (n_samples, n_chains) }
    """
    # Parse index file — maps variable name → (start_row, end_row)
    index_path = os.path.join(workdir, "CODAindex.txt")
    var_index = {}                          # name → (start, end)   1-indexed rows
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^"?([^"]+)"?\s+(\d+)\s+(\d+)$', line)
            if m:
                name, start, end = m.group(1), int(m.group(2)), int(m.group(3))
                var_index[name] = (start, end)

    # Read each chain file into a 2-D array (row, col) where col 0 = iter, col 1 = value
    chain_rows = []
    for c in range(1, n_chains + 1):
        chain_path = os.path.join(workdir, f"CODAchain{c}.txt")
        chain_rows.append(np.loadtxt(chain_path))  # shape (total_rows, 2)

    # Extract requested monitors
    result = {}
    for mon in monitors:
        # scalar: name matches exactly
        if mon in var_index:
            # start:end are 1-indexed row numbers in the CODA chain file
            # For a scalar monitored over n_samples iterations: start=1, end=n_samples
            start, end = var_index[mon]
            rows = slice(start - 1, end)
            arr = np.column_stack([chain_rows[c][rows, 1] for c in range(n_chains)])
            result[mon] = arr                    # shape (n_samples_per_chain, n_chains)
        else:
            # Try vector: mon[1], mon[2], ...
            keys = sorted([k for k in var_index if k.startswith(f"{mon}[")],
                          key=lambda s: int(re.search(r'\[(\d+)\]', s).group(1)))
            if not keys:
                continue
            slices = []
            for k in keys:
                start, end = var_index[k]
                rows = slice(start - 1, end)
                slices.append(
                    np.column_stack([chain_rows[c][rows, 1] for c in range(n_chains)])
                )
            result[mon] = np.stack(slices, axis=0)  # (n_elements, n_samples, n_chains)

    return result


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #
def run_jags(
    model_file,
    data,
    inits,
    monitors,
    n_adapt   = 1000,
    n_burnin  = 2000,
    n_samples = 2000,
    n_chains  = 4,
    workdir   = None,
    verbose   = True,
):
    """
    Run JAGS and return posterior samples.

    Parameters
    ----------
    model_file : str
        Path to the .jags model file.
    data : dict
        Data to pass (Python scalars or numpy arrays).
    inits : list of dict
        One init dict per chain.
    monitors : list of str
        Variables to monitor.
    n_adapt, n_burnin, n_samples : int
    n_chains : int
    workdir : str or None
        Working directory for JAGS files.  Uses a temp dir if None.
    verbose : bool

    Returns
    -------
    dict  { varname : ndarray (n_samples, n_chains) }
    """
    own_workdir = workdir is None
    if own_workdir:
        workdir = tempfile.mkdtemp(prefix="jags_")

    try:
        model_abs = os.path.abspath(model_file)
        data_path = os.path.join(workdir, "data.R")
        write_r_dump(data_path, data)

        init_paths = []
        for i, init in enumerate(inits):
            p = os.path.join(workdir, f"inits{i+1}.R")
            write_r_dump(p, init)
            init_paths.append(p)

        # Build JAGS script
        monitor_cmds = "\n".join(f'monitor {m}' for m in monitors)
        init_cmds    = "\n".join(
            f'inits in "{p}"' for p in init_paths
        )
        script = textwrap.dedent(f"""\
            model in "{model_abs}"
            data  in "{data_path}"
            compile, nchains({n_chains})
            {init_cmds}
            initialize
            adapt  {n_adapt}
            update {n_burnin}
            {monitor_cmds}
            update {n_samples}
            coda *
        """)
        script_path = os.path.join(workdir, "run.jags")
        with open(script_path, "w") as f:
            f.write(script)

        if verbose:
            print(f"Running JAGS in {workdir} ...")

        result = subprocess.run(
            ["jags", script_path],
            cwd=workdir,
            capture_output=not verbose,
            text=True,
        )
        if result.returncode != 0:
            if not verbose:
                print(result.stdout)
                print(result.stderr)
            raise RuntimeError(f"JAGS exited with code {result.returncode}")

        samples = _parse_coda(workdir, monitors, n_chains, n_samples)

    finally:
        if own_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    return samples
