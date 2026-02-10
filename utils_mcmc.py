from numpyro.infer import Predictive
import jax
import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import jax, jax.numpy as jnp
import numpyro
from functools import partial



import warnings
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import xarray as xr


def _get_sample_stat(idata, *names):
    """Return sample_stats[name] as a NumPy array if present, else None."""
    if not hasattr(idata, "sample_stats") or idata.sample_stats is None:
        return None
    ss = idata.sample_stats
    for name in names:
        if name in ss:
            return np.asarray(ss[name])
    return None


def _scalar_var_names(idata, max_vars=12):
    """Pick scalar parameter names (no extra dims beyond chain/draw)."""
    names = []
    if not hasattr(idata, "posterior") or idata.posterior is None:
        return names
    for name, da in idata.posterior.data_vars.items():
        # posterior dims typically: ("chain", "draw", ...) -> scalar if no extra dims
        if da.ndim == 2:
            names.append(name)
    # Cap the list to avoid gigantic plots
    return names[:max_vars]


def _figure_from_return(ret):
    """ArviZ plotting functions return axes arrays; get the figure."""
    if isinstance(ret, (list, tuple, np.ndarray)):
        # find first axes that has a figure
        for x in np.ravel(ret):
            try:
                return x.figure
            except Exception:
                continue
        return plt.gcf()
    # single axes
    try:
        return ret.figure
    except Exception:
        return plt.gcf()


def nuts_pdf_report(
    idata: az.InferenceData,
    out_path: str = "nuts_report.pdf",
    *,
    model_name: str | None = None,
    var_names: list[str] | None = None,
    pair_vars: list[str] | None = None,
    max_summary_rows: int = 40,
    rank_kind: str = "bars",  # or "vlines"
    ess_kind: str = "local",  # "local" or "bulk"
    rc_params: dict | None = None,
) -> Path:
    """
    Create a multi-page PDF with NUTS diagnostics from an ArviZ InferenceData object.

    Parameters
    - idata: az.InferenceData from az.from_numpyro(mcmc) or similar
    - out_path: output PDF path
    - model_name: optional title
    - var_names: which variables to show in summary/plots (defaults to scalar vars)
    - pair_vars: small subset (2–5 vars) for pair plot with divergences highlighted
    - max_summary_rows: truncate summary table for readability
    - rank_kind: 'bars' or 'vlines' for az.plot_rank
    - ess_kind: 'local' or 'bulk' for az.plot_ess
    - rc_params: optional matplotlib rcParams updates for aesthetics

    Returns
    - Path to the written PDF
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if rc_params:
        plt.rcParams.update(rc_params)

    # Determine defaults for variable selections
    if var_names is None:
        var_names = _scalar_var_names(idata, max_vars=12)

    if pair_vars is None:
        # choose up to 4 scalar variables for pair plot
        pair_vars = var_names[:4]

    # Compute core diagnostics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = az.summary(
            idata,
            var_names=var_names if var_names else None,
            round_to=3,
            kind="all",
        )

    # Sampler stats
    divergences = _get_sample_stat(idata, "diverging")
    accept = _get_sample_stat(idata, "acceptance_rate", "accept_prob", "accept_ratio")
    tree_depth = _get_sample_stat(idata, "tree_depth")
    n_steps = _get_sample_stat(idata, "n_steps", "num_steps")
    step_size = _get_sample_stat(idata, "step_size")
    energy = _get_sample_stat(idata, "energy")
    bfmi = None
    try:
        bfmi = az.bfmi(idata)  # returns per-chain BFMI
    except Exception:
        pass

    # Title and metadata
    title = "NUTS Diagnostics"
    if model_name:
        title += f" — {model_name}"

    # Create PDF
    with PdfPages(out_path) as pdf:
        # 1) Title and high-level stats page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        lines = [title, ""]
        try:
            nchains = idata.posterior.sizes.get("chain", None)
            ndraws = idata.posterior.sizes.get("draw", None)
        except Exception:
            nchains = ndraws = None

        if nchains and ndraws:
            lines.append(f"Chains: {nchains}   Draws per chain (post-warmup): {ndraws}")

        if divergences is not None:
            div_frac = float(np.mean(divergences))
            lines.append(f"Divergences: {int(div_frac * divergences.size)} / {divergences.size} ({div_frac:.2%})")

        if accept is not None:
            lines.append(f"Mean acceptance probability: {np.nanmean(accept):.3f}")

        if step_size is not None:
            # step_size may be scalar per chain or per draw; summarize
            ss_mean = np.nanmean(step_size)
            ss_sd = np.nanstd(step_size)
            lines.append(f"Step size (mean ± sd): {ss_mean:.3g} ± {ss_sd:.3g}")

        if tree_depth is not None:
            lines.append(f"Mean tree depth: {np.nanmean(tree_depth):.2f}")

        if n_steps is not None:
            lines.append(f"Mean NUTS steps per draw: {np.nanmean(n_steps):.2f}")

        if bfmi is not None:
            bfmi_vals = np.asarray(bfmi)
            lines.append(f"E-BFMI per chain: {np.array2string(bfmi_vals, precision=3)}  (min={np.min(bfmi_vals):.3f})")

        if energy is not None:
            lines.append(f"Energy var: {np.nanvar(energy):.3f}")

        text = "\n".join(lines)
        ax.text(0.05, 0.98, text, va="top", ha="left", fontsize=12)
        pdf.savefig(fig); plt.close(fig)

        # 2) Summary table (possibly split across pages)
        if isinstance(summary, pd.DataFrame) and len(summary) > 0:
            df = summary.copy()
            if len(df) > max_summary_rows:
                warnings.warn(
                    f"Summary has {len(df)} rows; truncating to first {max_summary_rows}."
                )
            df = df.head(max_summary_rows)
            # Render as a matplotlib table
            nrows = len(df)
            fig_height = max(3.5, 1.0 + 0.25 * nrows)
            fig, ax = plt.subplots(figsize=(11, fig_height))
            ax.axis("off")
            tbl = ax.table(
                cellText=np.round(df.values, 3).astype(object),
                colLabels=df.columns.tolist(),
                rowLabels=df.index.tolist(),
                loc="center",
                cellLoc="right",
                rowLoc="left",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.2)
            ax.set_title("Posterior summary (mean, sd, hdi, ess_bulk/tail, r_hat)", pad=10)
            pdf.savefig(fig); plt.close(fig)

        # 3) Trace plot (compact)
        if var_names:
            ret = az.plot_trace(idata, var_names=var_names, compact=True, kind="rank_vlines")
            fig = _figure_from_return(ret)
            fig.suptitle("Trace plots (compact)", y=1.02)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # 4) R-hat plot
        #if var_names:
        #    ret = az.plot_rhat(idata, var_names=var_names)
        #    fig = _figure_from_return(ret)
        #    fig.suptitle("R-hat", y=1.02)
        #    fig.tight_layout()
        #    pdf.savefig(fig); plt.close(fig)

        # 5) ESS plot
        if var_names:
            ret = az.plot_ess(idata, var_names=var_names, kind=ess_kind)
            fig = _figure_from_return(ret)
            fig.suptitle(f"Effective sample size ({ess_kind})", y=1.02)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # 6) Energy and E-BFMI
        if energy is not None:
            ret = az.plot_energy(idata)
            fig = _figure_from_return(ret)
            fig.suptitle("Energy plot (E-BFMI check)", y=1.02)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # 7) Autocorrelation
        if var_names:
            ret = az.plot_autocorr(idata, var_names=var_names)
            fig = _figure_from_return(ret)
            fig.suptitle("Autocorrelation", y=1.02)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # 8) Rank plots
        if var_names:
            ret = az.plot_rank(idata, var_names=var_names, kind=rank_kind)
            fig = _figure_from_return(ret)
            fig.suptitle("Rank plots", y=1.02)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # 9) Treedepth histogram
        if tree_depth is not None:
            fig, ax = plt.subplots(figsize=(7, 5))
            flat_td = tree_depth.flatten()
            ax.hist(flat_td[~np.isnan(flat_td)], bins=range(int(np.nanmin(flat_td)), int(np.nanmax(flat_td)) + 2), color="#4477AA", edgecolor="white")
            ax.set_title("Tree depth histogram")
            ax.set_xlabel("tree_depth")
            ax.set_ylabel("count")
            pdf.savefig(fig); plt.close(fig)

        # 10) NUTS steps per draw histogram
        if n_steps is not None:
            fig, ax = plt.subplots(figsize=(7, 5))
            vals = n_steps.flatten()
            ax.hist(vals[~np.isnan(vals)], bins=30, color="#66CC88", edgecolor="white")
            ax.set_title("NUTS number of steps per draw")
            ax.set_xlabel("n_steps")
            ax.set_ylabel("count")
            pdf.savefig(fig); plt.close(fig)

        # 11) Pair plot with divergences highlighted (if few vars)
        if len(pair_vars) >= 2:
            try:
                ret = az.plot_pair(
                    idata,
                    var_names=pair_vars,
                    marginals=True,
                    kind="kde",
                    divergences="bottom",  # highlight divergences if available
                )
                fig = _figure_from_return(ret)
                fig.suptitle(f"Pair plot ({', '.join(pair_vars)})", y=1.02)
                fig.tight_layout()
                pdf.savefig(fig); plt.close(fig)
            except Exception as e:
                warnings.warn(f"Pair plot failed: {e}")

    return out_path


# ---------------------------
# Example usage with NumPyro:
# ---------------------------
if __name__ == "__main__":
    # This example assumes you have run NumPyro MCMC with NUTS and have `idata`.
    # If you want a minimal runnable example, uncomment the block below.

    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    def toy_model():
        mu = numpyro.sample("mu", dist.Normal(0, 1))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1))
        with numpyro.plate("N", 200):
            numpyro.sample("y", dist.Normal(mu, sigma), obs=jnp.array(jax.random.normal(jax.random.PRNGKey(0), (200,))))

    nuts = NUTS(toy_model)
    mcmc = MCMC(nuts, num_warmup=1000, num_samples=1000, num_chains=4, chain_method="parallel")
    mcmc.run(jax.random.PRNGKey(1))
    mcmc.print_summary()

    # Convert to ArviZ InferenceData
    idata = az.from_numpyro(mcmc)

    # Generate report
    out = nuts_pdf_report(idata, out_path="nuts_report.pdf", model_name="toy_model")
    print(f"Wrote {out.resolve()}")
    """
    pass

