import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro as npyr
import numpyro.distributions as dist
from numpyro.infer import SVI, RenyiELBO, Predictive
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
from numpyro.infer.reparam import LocScaleReparam
from numpyro.ops.indexing import Vindex
from numpyro import handlers
from numpyro.optim import Adam

# ---------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------
def logistic_mvnorm(M=None, L=None, y=None):
    """Logistic mixed model with Gaussian random effects.

    G = s * (L @ z) where L is the Cholesky factor of the genetic
    relationship matrix and z ~ Normal(0, 1).
    """
    p = jnp.mean(y)
    beta0 = numpyro.sample("beta0", dist.Normal(jnp.log(p / (1 - p)), 5.0))
    s = numpyro.sample("s", dist.HalfNormal(5.0))
    mu = numpyro.deterministic("mu", 0.5 * s**2)
    with numpyro.plate("M_individuals", M):
        z = numpyro.sample("z", dist.Normal(0, 1))
    G = s * jnp.matmul(L, z)
    numpyro.sample("y", dist.Bernoulli(logits=beta0 + G), obs=y)


def logistic_mix2_mvnorm(M=None, L=None, y=None, priorscale=5.0, priorloc=0.5, prior_fam="halfnormal"):
    # L is Cholesky decomposition of correlation matrix of mixture distribution
    # mixture distribution has mixing proportions [1 - p, p], class-conditional means [-mu, mu],
    p = jnp.mean(y)
    beta0 = npyr.sample("beta0", dist.Normal(jnp.log(p / (1 - p)), 5))
    # class-conditional distributions of log likelihood ratio have means +/- mu and scale sqrt(2 * mu)
    if prior_fam == "lognormal":
        s = npyr.sample("s", dist.LogNormal(loc=priorloc, scale=priorscale))
    else:
        s = npyr.sample("s", dist.HalfNormal(priorscale))
    mu = npyr.deterministic("mu", 0.5 * s**2) # expected log-likelihood ratio

    component_loc = jnp.stack([-mu, mu], axis=0)
    component_scale = jnp.stack([s, s], axis=0)
    mixing_dist = dist.Categorical(probs=jnp.array([1 - p, p]))
    component_dist = dist.Normal(loc = component_loc, scale = component_scale)

    with npyr.plate(name="M individuals", size=M):
        W = npyr.sample("W", dist.MixtureSameFamily(mixing_dist, component_dist))
    Z = W - jnp.mean(W)
    G = jnp.matmul(L, Z)
    npyr.sample('y', dist.Bernoulli(logits=beta0 + G), obs=y)
    return(logistic_mix2_mvnorm)


def logistic_mix2_conditioned(M=None, L=None, y=None, priorscale=5.0, priorloc=0.5, prior_fam="halfnormal"):
    """Mixture model with class assignments fixed to observed disease status.

    Identical to logistic_mix2_mvnorm except the latent mixture indicator
    for each individual is set equal to the observed y, so that cases draw
    from the +mu component and controls from the -mu component.  This
    eliminates the 2^M discrete multimodality that prevents NUTS from
    mixing in the unconditioned mixture model.
    """
    p = jnp.mean(y)
    beta0 = npyr.sample("beta0", dist.Normal(jnp.log(p / (1 - p)), 5))
    if prior_fam == "lognormal":
        s = npyr.sample("s", dist.LogNormal(loc=priorloc, scale=priorscale))
    else:
        s = npyr.sample("s", dist.HalfNormal(priorscale))
    mu = npyr.deterministic("mu", 0.5 * s**2)

    # Condition mixture component on observed y:
    #   y=1 (case)    -> W ~ N(+mu, s²)
    #   y=0 (control) -> W ~ N(-mu, s²)
    loc = (2 * y - 1) * mu

    with npyr.plate(name="M individuals", size=M):
        W = npyr.sample("W", dist.Normal(loc, s))
    Z = W - jnp.mean(W)
    G = jnp.matmul(L, Z)
    npyr.sample('y', dist.Bernoulli(logits=beta0 + G), obs=y)


def lr_discrete(M=None, L=None, y=None, muprior=dist.Gamma, priorscale=0.75, priorloc=2.0):
    """Logistic mixed model with explicit discrete class-membership indicators.

    Uses config_enumerate for analytic marginalization of discrete z during
    NUTS, and LocScaleReparam(centered=0) to break the funnel between mu
    and the individual-level latent variables.
    """
    assert M is not None and L is not None and y is not None
    assert y.shape[0] == M and L.shape == (M, M)

    p = jnp.mean(y)
    K = 20
    p = numpyro.sample("p", dist.Beta(K * p, K * (1. - p)))
    beta0 = jnp.log(p / (1.0 - p))

    mu = numpyro.sample("mu", muprior(priorloc, priorscale))
    s = numpyro.deterministic("s", jnp.sqrt(2.0 * mu))

    component_loc = jnp.stack([-mu, mu], axis=0)
    component_scale = jnp.stack([s, s], axis=0)
    mixing_probs = jnp.array([1.0 - p, p])

    with numpyro.plate("M individuals", M, dim=-1):
        z = numpyro.sample("z", dist.Categorical(probs=mixing_probs),
                           infer={"enumerate": "sequential"})

        loc_z = Vindex(component_loc)[z]
        scale_z = Vindex(component_scale)[z]

        with handlers.reparam(config={"Z_mix": LocScaleReparam(centered=0)}):
            Z_mix = numpyro.sample("Z_mix", dist.Normal(loc=loc_z, scale=scale_z))

    Z = Z_mix - (2.0 * p - 1.0) * mu
    G = numpyro.deterministic("G", jnp.matmul(Z, L.T))
    numpyro.sample("y", dist.Bernoulli(logits=beta0 + G), obs=y)


def lr_discrete_blockdiag(L_blocks=None, y_flat=None, n_blocks=None, block_size=None,
                           p_obs=None, muprior=dist.Gamma, priorscale=0.75, priorloc=2.0):
    """Block-diagonal variant of lr_discrete for relatives-only data.

    Operates on batched Cholesky factors of uniform-size family blocks,
    using einsum for efficient batched matmul instead of a single large
    M x M matrix multiply.

    Parameters
    ----------
    L_blocks : jnp.ndarray, shape (n_blocks, block_size, block_size)
        Per-block Cholesky factors of the genetic relationship sub-matrices.
    y_flat : jnp.ndarray, shape (n_blocks * block_size,)
        Flattened binary outcomes, block-contiguous.
    n_blocks : int
        Number of family blocks.
    block_size : int
        Uniform size of each block.
    p_obs : float
        Observed case proportion in the full sample (for the Beta prior).
    muprior, priorscale, priorloc :
        Prior family and hyperparameters for mu (same as lr_discrete).
    """
    M = n_blocks * block_size

    K = 20
    p = numpyro.sample("p", dist.Beta(K * p_obs, K * (1.0 - p_obs)))
    beta0 = jnp.log(p / (1.0 - p))

    mu = numpyro.sample("mu", muprior(priorloc, priorscale))
    s = numpyro.deterministic("s", jnp.sqrt(2.0 * mu))

    component_loc = jnp.stack([-mu, mu], axis=0)
    component_scale = jnp.stack([s, s], axis=0)
    mixing_probs = jnp.array([1.0 - p, p])

    with numpyro.plate("individuals", M, dim=-1):
        z = numpyro.sample("z", dist.Categorical(probs=mixing_probs),
                           infer={"enumerate": "sequential"})

        loc_z = Vindex(component_loc)[z]
        scale_z = Vindex(component_scale)[z]

        with handlers.reparam(config={"Z_mix": LocScaleReparam(centered=0)}):
            Z_mix = numpyro.sample("Z_mix", dist.Normal(loc=loc_z, scale=scale_z))

    Z = Z_mix - (2.0 * p - 1.0) * mu
    # Batched block matmul: Z @ L.T per block via einsum
    Z_blocks = Z.reshape(n_blocks, block_size)
    G_blocks = jnp.einsum('bi,bji->bj', Z_blocks, L_blocks)
    G = numpyro.deterministic("G", G_blocks.reshape(-1))

    numpyro.sample("y", dist.Bernoulli(logits=beta0 + G), obs=y_flat)


# ---------------------------------------------------------------------
# Block-diagonal preprocessing
# ---------------------------------------------------------------------
def find_blocks(A):
    """Find connected components (families) in the genetic relationship matrix.

    Uses off-diagonal structure of A to identify groups of related individuals.

    Parameters
    ----------
    A : ndarray, shape (M, M)
        Genetic relationship matrix.

    Returns
    -------
    blocks : list of ndarray
        Each element contains row indices of one connected component,
        sorted by block size descending.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    M = A.shape[0]
    off_diag = np.abs(A) - np.eye(M)
    mask = off_diag > 1e-10
    n_comp, labels = connected_components(
        csr_matrix(mask.astype(float)), directed=False
    )
    blocks = [np.where(labels == c)[0] for c in range(n_comp)]
    blocks.sort(key=len, reverse=True)
    return blocks


def reduce_to_relatives(A, y, min_block_size=2):
    """Remove singletons, keeping only individuals with relatives in the sample.

    Constructs a reduced relationship matrix, its Cholesky factor, and
    outcome vector for individuals in connected components of size >= min_block_size.
    The returned arrays are ordered so that blocks are contiguous.

    Parameters
    ----------
    A : ndarray, shape (M, M)
        Genetic relationship matrix.
    y : ndarray, shape (M,)
        Binary outcome.
    min_block_size : int
        Minimum block size to retain.

    Returns
    -------
    y_red : ndarray, shape (M_red,)
    A_red : ndarray, shape (M_red, M_red)
    L_red : ndarray, shape (M_red, M_red)
    kept_idx : ndarray of int, shape (M_red,)
        Original indices of kept individuals.
    block_sizes : list of int
        Size of each retained block.
    """
    blocks = find_blocks(A)
    kept = [b for b in blocks if len(b) >= min_block_size]
    if not kept:
        empty = np.array([], dtype=int)
        return (np.array([]), np.empty((0, 0)), np.empty((0, 0)),
                empty, [])
    kept_idx = np.concatenate(kept)
    y_red = y[kept_idx]
    A_red = A[np.ix_(kept_idx, kept_idx)]
    L_red = np.linalg.cholesky(A_red)
    block_sizes = [len(b) for b in kept]
    return y_red, A_red, L_red, kept_idx, block_sizes


# ---------------------------------------------------------------------
# SVI fitting
# ---------------------------------------------------------------------
def fit_svi_lowrank(
    M, L, y,
    model=None,
    rank=20,
    n_steps=15000,
    lr=0.001,
    num_samples=2000,
    seed=0,
    progress_bar=True,
):
    """Fit logistic model via SVI with AutoLowRankMultivariateNormal guide.

    Parameters
    ----------
    M : int
        Number of individuals.
    L : jnp.ndarray, shape (M, M)
        Cholesky factor of the genetic relationship matrix.
    y : jnp.ndarray, shape (M,)
        Binary case/control status.
    model : callable, optional
        NumPyro model function with signature (M, L, y).
        Defaults to logistic_mvnorm.
    rank : int
        Rank of the low-rank covariance approximation.
    n_steps : int
        Number of SVI optimization steps.
    lr : float
        Adam learning rate.
    num_samples : int
        Number of posterior samples to draw from the fitted guide.
    seed : int
        Random seed.
    progress_bar : bool
        Whether to display a progress bar.

    Returns
    -------
    dict with keys:
        s, mu, beta0 : jnp.ndarray — posterior samples
        svi_result : numpyro SVI result object
        losses : jnp.ndarray — ELBO loss trace
    """
    if model is None:
        model = logistic_mvnorm

    rng_key = random.key(seed)
    guide = AutoLowRankMultivariateNormal(model, rank=rank)
    optimizer = Adam(step_size=lr)
    elbo = RenyiELBO(alpha=0, num_particles=2)
    svi = SVI(model, guide, optimizer, loss=elbo)

    svi_result = svi.run(
        rng_key, n_steps, M=M, L=L, y=y, progress_bar=progress_bar
    )

    params = svi_result.params
    predictive = Predictive(guide, params=params, num_samples=num_samples)
    posterior_samples = predictive(random.PRNGKey(seed + 1))

    s_samples = posterior_samples["s"]
    mu_samples = 0.5 * s_samples**2
    beta0_samples = posterior_samples["beta0"]

    return dict(
        s=s_samples,
        mu=mu_samples,
        beta0=beta0_samples,
        svi_result=svi_result,
        losses=svi_result.losses,
    )


# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Simulation: case-control with genetic relationship matrix
# ---------------------------------------------------------------------
def simulate_casecontrol_related(
    n_fullsib_pairs=100_000,
    n_halfsib_pairs=10_000,
    n_unrelated=1000,
    K=0.01,
    mu=1.0,
    seed=42,
    return_genotypic_values=False,
):
    """Simulate case-control data from a population with related individuals.

    Generates genotypic values from a multivariate Gaussian whose covariance
    structure is determined by a genetic relationship matrix containing
    full-sib pairs (correlation 0.5), half-sib pairs (correlation 0.25),
    and unrelated individuals.  Disease status follows a logistic model
    P(y=1|g) = expit(alpha + g), with alpha calibrated to target prevalence K.

    The case-control sample includes all cases and an equal number of randomly
    sampled controls.  The returned relationship matrix and its Cholesky factor
    correspond to the sampled individuals only.

    Note: Under the logistic link, lambda_S < exp(mu) because the logistic
    function saturates for high-g individuals who dominate E[risk1*risk2].
    The approximation lambda_S ≈ exp(mu) is good for mu < ~1 (ratio > 0.9)
    but deteriorates as mu increases.

    Parameters
    ----------
    n_fullsib_pairs : int
        Number of full-sib pairs (genetic correlation 0.5).
        Default 100000 yields ~25 concordant-affected pairs at K=0.01, mu=1.
    n_halfsib_pairs : int
        Number of half-sib pairs (genetic correlation 0.25).
    n_unrelated : int
        Number of unrelated singletons.
    K : float
        Target population disease prevalence.
    mu : float
        Genetic information for discrimination (nats).  Marginal variance
        of genotypic values is 2*mu.
    seed : int
        Random seed.
    return_genotypic_values : bool
        If True, return the genotypic values for the sampled individuals
        as a fifth element of the returned tuple.

    Returns
    -------
    y_sample : ndarray of int, shape (M,)
        Case/control status (1/0) for the sampled individuals.
    A_sample : ndarray, shape (M, M)
        Genetic relationship matrix for the sampled individuals.
    L_sample : ndarray, shape (M, M)
        Cholesky factor of A_sample.
    info : dict
        Simulation parameters and diagnostics including empirical lambda_S.
    g_sample : ndarray of float, shape (M,)
        Genotypic values for the sampled individuals (only if
        return_genotypic_values is True).
    """
    from scipy.optimize import brentq
    from scipy.special import expit

    rng = np.random.default_rng(seed)
    sigma = np.sqrt(2 * mu)
    N = 2 * n_fullsib_pairs + 2 * n_halfsib_pairs + n_unrelated

    # --- Generate genotypic values per family type ---
    # Full-sib pairs: bivariate normal, correlation 0.5
    z1 = rng.standard_normal(n_fullsib_pairs)
    z2 = rng.standard_normal(n_fullsib_pairs)
    g_fs1 = sigma * z1
    g_fs2 = sigma * (0.5 * z1 + np.sqrt(0.75) * z2)

    # Half-sib pairs: bivariate normal, correlation 0.25
    z1 = rng.standard_normal(n_halfsib_pairs)
    z2 = rng.standard_normal(n_halfsib_pairs)
    g_hs1 = sigma * z1
    g_hs2 = sigma * (0.25 * z1 + np.sqrt(1 - 0.0625) * z2)

    # Unrelated individuals
    g_unrel = sigma * rng.standard_normal(n_unrelated)

    # Layout: [fs_pair0_a, fs_pair0_b, fs_pair1_a, ..., hs_pair0_a, hs_pair0_b, ..., unrel...]
    g = np.concatenate([
        np.column_stack([g_fs1, g_fs2]).ravel(),
        np.column_stack([g_hs1, g_hs2]).ravel(),
        g_unrel,
    ])

    # --- Find intercept alpha for target prevalence K ---
    alpha = brentq(lambda a: np.mean(expit(a + g)) - K, -30, 10)

    # --- Simulate disease status ---
    y = rng.binomial(1, expit(alpha + g)).astype(int)
    observed_K = y.mean()
    print(y.shape)

    # --- Empirical lambda_S from full-sib pairs ---
    y_fs = y[: 2 * n_fullsib_pairs].reshape(n_fullsib_pairs, 2)
    proband_aff_1 = y_fs[:, 0] == 1
    proband_aff_2 = y_fs[:, 1] == 1
    n_probands = proband_aff_1.sum() + proband_aff_2.sum()
    if n_probands > 0:
        recurrence = (
            y_fs[proband_aff_1, 1].sum() + y_fs[proband_aff_2, 0].sum()
        ) / n_probands
        empirical_lambda_S = recurrence / observed_K
    else:
        empirical_lambda_S = np.nan

    n_concordant_case_fs = int((y_fs.sum(axis=1) == 2).sum())

    # --- Case-control sampling: all cases + equal number of controls ---
    case_idx = np.where(y == 1)[0]
    control_idx = np.where(y == 0)[0]
    n_cases = len(case_idx)
    sampled_controls = rng.choice(control_idx, size=n_cases, replace=False)
    sample_idx = np.sort(np.concatenate([case_idx, sampled_controls]))
    y_sample = y[sample_idx]
    M = len(sample_idx)

    # --- Build genetic relationship matrix for sampled individuals ---
    A_sample = np.eye(M, dtype=np.float64)

    # Full-sib pairs: original indices (2k, 2k+1) for k in range(n_fullsib_pairs)
    fs_first = np.arange(0, 2 * n_fullsib_pairs, 2)
    fs_second = fs_first + 1
    pos_1 = np.clip(np.searchsorted(sample_idx, fs_first), 0, M - 1)
    pos_2 = np.clip(np.searchsorted(sample_idx, fs_second), 0, M - 1)
    both_in_fs = (sample_idx[pos_1] == fs_first) & (sample_idx[pos_2] == fs_second)

    i_fs = pos_1[both_in_fs]
    j_fs = pos_2[both_in_fs]
    A_sample[i_fs, j_fs] = 0.5
    A_sample[j_fs, i_fs] = 0.5
    n_sampled_fs_pairs = int(both_in_fs.sum())
    n_sampled_aff_fs_pairs = int(
        np.sum(both_in_fs & (y_fs[:, 0] == 1) & (y_fs[:, 1] == 1))
    )

    # Half-sib pairs: original indices (hs_offset+2k, hs_offset+2k+1)
    hs_offset = 2 * n_fullsib_pairs
    hs_first = np.arange(hs_offset, hs_offset + 2 * n_halfsib_pairs, 2)
    hs_second = hs_first + 1
    pos_1 = np.clip(np.searchsorted(sample_idx, hs_first), 0, M - 1)
    pos_2 = np.clip(np.searchsorted(sample_idx, hs_second), 0, M - 1)
    both_in_hs = (sample_idx[pos_1] == hs_first) & (sample_idx[pos_2] == hs_second)

    i_hs = pos_1[both_in_hs]
    j_hs = pos_2[both_in_hs]
    A_sample[i_hs, j_hs] = 0.25
    A_sample[j_hs, i_hs] = 0.25
    n_sampled_hs_pairs = int(both_in_hs.sum())

    L_sample = np.linalg.cholesky(A_sample)

    theoretical_lambda_S = np.exp(mu)

    info = dict(
        mu=mu,
        s=sigma,
        K_target=K,
        K_observed=observed_K,
        alpha=alpha,
        N=N,
        n_fullsib_pairs=n_fullsib_pairs,
        n_halfsib_pairs=n_halfsib_pairs,
        n_unrelated=n_unrelated,
        n_cases_pop=n_cases,
        n_concordant_case_fullsib_pop=n_concordant_case_fs,
        M=M,
        n_cases_sample=int(y_sample.sum()),
        n_controls_sample=int(M - y_sample.sum()),
        n_sampled_fullsib_pairs=n_sampled_fs_pairs,
        n_sampled_aff_fullsib_pairs=n_sampled_aff_fs_pairs,
        n_sampled_halfsib_pairs=n_sampled_hs_pairs,
        theoretical_lambda_S=theoretical_lambda_S,
        empirical_lambda_S=empirical_lambda_S,
    )

    if return_genotypic_values:
        g_sample = g[sample_idx]
        return y_sample, A_sample, L_sample, info, g_sample

    return y_sample, A_sample, L_sample, info


def print_casecontrol_summary(info):
    """Print a summary of the case-control simulation with lambda_S verification."""
    print("=== Population ===")
    print(f"  Total individuals:     {info['N']:>12,}")
    print(f"    Full-sib pairs:      {info['n_fullsib_pairs']:>12,}")
    print(f"    Half-sib pairs:      {info['n_halfsib_pairs']:>12,}")
    print(f"    Unrelated:           {info['n_unrelated']:>12,}")
    print(f"  Cases in population:   {info['n_cases_pop']:>12,}")
    print(f"  Observed prevalence:   {info['K_observed']:.6f}"
          f"  (target: {info['K_target']:.4f})")
    print(f"  Concordant-affected full-sib pairs: {info['n_concordant_case_fullsib_pop']}")

    print("\n=== Sibling recurrence risk ratio (lambda_S) ===")
    print(f"  Theoretical exp(mu):   {info['theoretical_lambda_S']:.4f}")
    print(f"  Empirical (full sibs): {info['empirical_lambda_S']:.4f}")
    ratio = info['empirical_lambda_S'] / info['theoretical_lambda_S']
    print(f"  Ratio empirical/theoretical: {ratio:.4f}")

    print("\n=== Case-control sample ===")
    print(f"  Sample size M:         {info['M']:>6}")
    print(f"    Cases:               {info['n_cases_sample']:>6}")
    print(f"    Controls:            {info['n_controls_sample']:>6}")
    print(f"  Full-sib pairs (both sampled):  {info['n_sampled_fullsib_pairs']}")
    print(f"    of which both affected:       {info['n_sampled_aff_fullsib_pairs']}")
    print(f"  Half-sib pairs (both sampled):  {info['n_sampled_halfsib_pairs']}")

    print("\n=== True parameters ===")
    print(f"  mu       = {info['mu']:.4f}  (expected log-LR, nats)")
    print(f"  s        = {info['s']:.4f}")
    print(f"  alpha    = {info['alpha']:.4f}  (intercept)")
    print(f"  K        = {info['K_target']:.4f}")



from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.layout import Layout


def create_data(seed, K, lambdaS):
    # ---------------------------------------------------------------------
    # 1. User-adjustable settings
    # ---------------------------------------------------------------------
    SEED        = seed          # for reproducibility
    N_PAIRS     = 80_000     # how many sibling pairs in the population
    POP_K       = K       # prevalence in the general population
    SIB_K       = lambdaS * K     # recurrence risk in sibs of a case
    # ---------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    
    # ---------------------------------------------------------------------
    # 2. Simulate the phenotypes for N_PAIRS sibling pairs
    #    (very simple conditional model: first sib = "proband")
    # ---------------------------------------------------------------------
    y1 = rng.random(N_PAIRS) < POP_K                 # proband status
    y2 = rng.random(N_PAIRS) < np.where(y1, SIB_K, POP_K)   # second sib
    console = Console()

    table = Table(title="Data simulation target and achieved")
    table.add_column("")
    table.add_column("")
    table.add_row("Target log Sibling risk", f"{np.log(lambdaS):.2f}")
    table.add_row("Empirical sample log Sibling risk", f"{np.log(y2[y1==1].mean()/y1.mean()):.2f}")
    table1 = table

    table = Table(title="Disease status in simulated data")
    table.add_column("")
    table.add_column("Proband control")
    table.add_column("Proband case")
    table.add_column("Total")
    table.add_row("Sibling control", f"{((y1 == 0) & (y2 == 0)).sum()}", f"{((y1 == 1) & (y2 == 0)).sum()}", f"{((y2 == 0)).sum()}")
    table.add_row("Sibling case", f"{((y1 == 0) & (y2 == 1)).sum()}", f"{((y1 == 1) & (y2 == 1)).sum()}", f"{((y2 == 1)).sum()}")
    table.add_row("Total", f"{((y1 == 0)).sum()}", f"{((y1 == 1)).sum()}", f"{((y2 == 1)|(y2==0)).sum()}")

    table2 = table
    emplogs = np.log(y2[y1==1].mean()/y1.mean())
    # stack the individuals
    y_all          = np.concatenate([y1, y2]).astype(int)   # (2*N_PAIRS,)
    fam_id_all     = np.concatenate([np.arange(N_PAIRS), 
                                    np.arange(N_PAIRS)]) 
    within_pair_ix = np.tile([0, 1], N_PAIRS)               # 0,1,0,1,...
    
    # ---------------------------------------------------------------------
    # 3. Case/control selection: keep every case + equal # controls
    # ---------------------------------------------------------------------
    case_idx      = np.where(y_all == 1)[0]
    n_cases       = len(case_idx)
    
    ctrl_idx_pool = np.where(y_all == 0)[0]
    ctrl_idx      = rng.choice(ctrl_idx_pool, size=n_cases, replace=False)
    
    keep_idx      = np.sort(np.concatenate([case_idx, ctrl_idx]))  # final order
    y             = y_all[keep_idx]
    fam_id        = fam_id_all[keep_idx]
    
    M = len(keep_idx)        # total sample size (cases + controls)
    # ---------------------------------------------------------------------
    # 4. Build the M × M genetic correlation matrix
    # ---------------------------------------------------------------------
    G = np.eye(M, dtype=float)                # start with the diagonal = 1
    H = -np.ones_like(G, dtype=float)                # start with the diagonal = 1
    # pairs that belong to the same family get correlation 0.5
    # (makes block-diagonal 2×2 sub-matrices)
    y1 = []
    y2 = []
    for fid in np.unique(fam_id):
        members = np.where(fam_id == fid)[0]
        if len(members) == 2:                 # a complete sibling pair selected
            i, j = members                    # unpack the two indices
            G[i, j] = G[j, i] = 0.5
            H[i, j] = H[j, i] = y[i] + y[j]
            y1.append(y[i])
            y2.append(y[j])

    y1 = np.array(y1)
    y2 = np.array(y2)
     
    table = Table(title=f"Simulated cases-control: Sample size  M = {M}  ({n_cases} cases, {n_cases} controls)\nDistribution of cases in siblings")
    table.add_column("")
    table.add_column("Proband control")
    table.add_column("Proband case")
    table.add_column("Total")
    table.add_row("Sibling control", f"{((y1 == 0) & (y2 == 0)).sum()}", f"{((y1 == 1) & (y2 == 0)).sum()}", f"{((y2 == 0)).sum()}")
    table.add_row("Sibling case", f"{((y1 == 0) & (y2 == 1)).sum()}", f"{((y1 == 1) & (y2 == 1)).sum()}", f"{((y2 == 1)).sum()}")
    table.add_row("Total", f"{((y1 == 0)).sum()}", f"{((y1 == 1)).sum()}", f"{((y2 == 1)|(y2==0)).sum()}")
    table3 = table
    console.print(Panel(Group(Columns([table1, table2, table3], expand=True)), title="Data simulation"))

    # put add a constant to those who have the same phenotype
    # ---------------------------------------------------------------------
    # 5. Cholesky factor (lower triangular)
    # ---------------------------------------------------------------------
    L = np.linalg.cholesky(G)                 # NumPy returns lower-triangular L
    GG = L @ L.T
    return M, L, y, y, emplogs


# ---------------------------------------------------------------------
# Main: quick simulation test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    y, A, L, info = simulate_casecontrol_related(
        n_fullsib_pairs=100_000, n_halfsib_pairs=10_000, n_unrelated=1000,
    )
    print_casecontrol_summary(info)
