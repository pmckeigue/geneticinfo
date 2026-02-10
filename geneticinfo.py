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
# Simulation: sib-pair case-control dataset
# ---------------------------------------------------------------------
def simulate_sibpair_data(
    n_pairs_pop=2_000_000,
    K=0.01,
    mu_true=1.0,
    n_concordant_case=None,
    n_discordant=1000,
    n_concordant_control=500,
    seed=42,
):
    """Simulate a sib-pair case-control dataset.

    Generates a large population of sib pairs whose joint disease status
    follows the marginal and conditional probabilities implied by:

      - population prevalence  K
      - sibling recurrence risk ratio  lambda_S = exp(mu_true)

    Complete sib pairs are then sampled by concordance type (both
    affected, discordant, both unaffected) so that the dataset is
    informative for estimating mu.

    Parameters
    ----------
    n_pairs_pop : int
        Number of sib pairs in the simulated population.
    K : float
        Population disease prevalence.
    mu_true : float
        True expected log-likelihood ratio (nats).  Equivalently,
        log(lambda_S) where lambda_S is the sibling recurrence risk ratio.
    n_concordant_case : int or None
        Concordant affected pairs to retain (None keeps all available).
    n_discordant : int
        Discordant pairs to retain.
    n_concordant_control : int
        Concordant unaffected pairs to retain.
    seed : int
        Random seed.

    Returns
    -------
    y : ndarray of int, shape (M,)
        Case/control status (1/0).
    C : ndarray, shape (M, M)
        Genetic relationship (correlation) matrix.
    L : ndarray, shape (M, M)
        Cholesky factor of C.
    info : dict
        True parameters and sample composition.
    """
    rng = np.random.default_rng(seed)

    lambda_S = np.exp(mu_true)
    s_true = np.sqrt(2.0 * mu_true)

    # --- Joint sib-pair probabilities ---
    #   P(y2=1 | y1=1) = K * lambda_S
    #   P(y2=1 | y1=0) = K * (1 - K*lambda_S) / (1 - K)
    p_aff_given_aff = K * lambda_S
    p_aff_given_unaff = K * (1.0 - K * lambda_S) / (1.0 - K)

    assert 0.0 < p_aff_given_aff < 1.0, (
        f"K*lambda_S = {p_aff_given_aff:.4f} outside (0,1); "
        "lambda_S too large for this prevalence"
    )

    # --- Simulate population of sib pairs ---
    y1 = (rng.random(n_pairs_pop) < K).astype(np.int8)
    y2_prob = np.where(y1, p_aff_given_aff, p_aff_given_unaff)
    y2 = (rng.random(n_pairs_pop) < y2_prob).astype(np.int8)

    # Classify pairs
    pair_sum = y1 + y2
    idx_cc   = np.where(pair_sum == 2)[0]   # both affected
    idx_disc = np.where(pair_sum == 1)[0]   # one affected
    idx_ctrl = np.where(pair_sum == 0)[0]   # both unaffected

    n_cc_pop   = len(idx_cc)
    n_disc_pop = len(idx_disc)
    n_ctrl_pop = len(idx_ctrl)

    # --- Sample complete pairs by concordance type ---
    n_cc   = n_cc_pop if n_concordant_case is None else min(n_concordant_case, n_cc_pop)
    n_disc = min(n_discordant, n_disc_pop)
    n_ctrl = min(n_concordant_control, n_ctrl_pop)

    sampled_cc   = rng.choice(idx_cc,   size=n_cc,   replace=False)
    sampled_disc = rng.choice(idx_disc, size=n_disc, replace=False)
    sampled_ctrl = rng.choice(idx_ctrl, size=n_ctrl, replace=False)

    sampled_pairs = np.sort(np.concatenate([sampled_cc, sampled_disc, sampled_ctrl]))
    n_pairs = len(sampled_pairs)

    # Interleave sibs: [sib1_pair0, sib2_pair0, sib1_pair1, sib2_pair1, ...]
    y = np.empty(2 * n_pairs, dtype=int)
    y[0::2] = y1[sampled_pairs]
    y[1::2] = y2[sampled_pairs]
    M = len(y)

    # --- Block-diagonal relationship matrix ---
    C = np.eye(M, dtype=np.float64)
    for k in range(n_pairs):
        i, j = 2 * k, 2 * k + 1
        C[i, j] = C[j, i] = 0.5
    L = np.linalg.cholesky(C)

    # --- Empirical lambda_S from the full population ---
    n_y1_aff = np.sum(y1 == 1)
    empirical_lambda_S = (np.mean(y2[y1 == 1]) / K) if n_y1_aff > 0 else np.nan

    n_cases = int(np.sum(y))
    n_controls = M - n_cases

    info = dict(
        mu_true=mu_true,
        s_true=s_true,
        lambda_S_true=lambda_S,
        K=K,
        n_pairs_pop=n_pairs_pop,
        pop_concordant_case=n_cc_pop,
        pop_discordant=n_disc_pop,
        pop_concordant_control=n_ctrl_pop,
        sample_concordant_case=n_cc,
        sample_discordant=n_disc,
        sample_concordant_control=n_ctrl,
        n_pairs=n_pairs,
        M=M,
        n_cases=n_cases,
        n_controls=n_controls,
        empirical_lambda_S=empirical_lambda_S,
    )
    return y, C, L, info


def print_sim_summary(info):
    """Print a concise summary of the simulated dataset."""
    print("=== Population ===")
    print(f"  Sib pairs simulated:   {info['n_pairs_pop']:>12,}")
    print(f"  Both affected:         {info['pop_concordant_case']:>12,}"
          f"  ({100 * info['pop_concordant_case'] / info['n_pairs_pop']:.4f}%)")
    print(f"  Discordant:            {info['pop_discordant']:>12,}"
          f"  ({100 * info['pop_discordant'] / info['n_pairs_pop']:.4f}%)")
    print(f"  Both unaffected:       {info['pop_concordant_control']:>12,}"
          f"  ({100 * info['pop_concordant_control'] / info['n_pairs_pop']:.4f}%)")
    print(f"  Empirical lambda_S:    {info['empirical_lambda_S']:.4f}"
          f"  (true: {info['lambda_S_true']:.4f})")

    print("\n=== Sampled pairs ===")
    print(f"  Both affected:         {info['sample_concordant_case']:>6}")
    print(f"  Discordant:            {info['sample_discordant']:>6}")
    print(f"  Both unaffected:       {info['sample_concordant_control']:>6}")
    print(f"  Total pairs:           {info['n_pairs']:>6}  →  {info['M']} individuals")
    print(f"  Cases: {info['n_cases']},  Controls: {info['n_controls']}")

    print("\n=== True parameters ===")
    print(f"  mu       = {info['mu_true']:.4f}  (expected log-LR, nats)")
    print(f"  s        = {info['s_true']:.4f}")
    print(f"  lambda_S = {info['lambda_S_true']:.4f}")
    print(f"  K        = {info['K']:.4f}")


# ---------------------------------------------------------------------
# Simulation: case-control with genetic relationship matrix
# ---------------------------------------------------------------------
def simulate_casecontrol_related(
    n_fullsib_pairs=100_000,
    n_halfsib_pairs=10_000,
    n_unrelated=100_000,
    K=0.01,
    mu=1.0,
    seed=42,
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


# ---------------------------------------------------------------------
# Main: quick simulation test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    y, A, L, info = simulate_casecontrol_related(
        n_fullsib_pairs=10_000, n_halfsib_pairs=1_000, n_unrelated=10_000,
    )
    print_casecontrol_summary(info)
