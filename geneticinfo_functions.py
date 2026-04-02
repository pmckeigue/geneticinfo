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

    mu_dist = muprior if isinstance(muprior, dist.Distribution) else muprior(priorloc, priorscale)
    mu = numpyro.sample("mu", mu_dist)
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


def lr_discrete_blockdiag(L_list=None, y_list=None, sizes=None, p_obs=None,
                           muprior=dist.HalfCauchy, priorscale=1.0):
    """Block-diagonal variant of lr_discrete for relatives-only data.

    Handles mixed block sizes (pairs, triplets, etc.) by grouping blocks
    by size and applying a batched einsum per group.

    Parameters
    ----------
    L_list : list of jnp.ndarray
        Per-size list of Cholesky factor arrays.  L_list[i] has shape
        (n_i, sizes[i], sizes[i]).  Groups should be in the same order as
        sizes and y_list; individuals are concatenated in that order.
    y_list : list of jnp.ndarray
        Per-size list of outcome vectors.  y_list[i] has shape
        (n_i * sizes[i],).
    sizes : list of int
        Block sizes corresponding to L_list / y_list entries.
    p_obs : float
        Observed case proportion in the full sample (for the Beta prior on p).
    muprior : numpyro distribution class or instance
        Prior for mu.  Defaults to HalfCauchy; pass an instantiated
        distribution to override.
    priorscale : float
        Scale parameter passed to muprior (ignored if muprior is already
        an instantiated distribution).
    """
    M = sum(L.shape[0] * s for L, s in zip(L_list, sizes))

    K = 20
    p = numpyro.sample("p", dist.Beta(K * p_obs, K * (1.0 - p_obs)))
    beta0 = jnp.log(p / (1.0 - p))

    mu_dist = muprior if isinstance(muprior, dist.Distribution) else muprior(priorscale)
    mu = numpyro.sample("mu", mu_dist)
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

    # Compute G for each block-size group; individuals are laid out
    # consecutively in the order given by L_list / sizes.
    G_parts = []
    offset = 0
    for L_s, block_size in zip(L_list, sizes):
        n_s = L_s.shape[0]
        n_inds = n_s * block_size
        Z_s = Z[offset:offset + n_inds].reshape(n_s, block_size)
        G_s = jnp.einsum('bi,bji->bj', Z_s, L_s).reshape(-1)
        G_parts.append(G_s)
        offset += n_inds

    G = numpyro.deterministic("G", jnp.concatenate(G_parts))
    numpyro.sample("y", dist.Bernoulli(logits=beta0 + G), obs=jnp.concatenate(y_list))


# ---------------------------------------------------------------------
# Conditional logistic likelihood
# ---------------------------------------------------------------------
def cond_loglik_blocks(G, y, block_sizes):
    """Conditional logistic regression log-likelihood summed over all blocks.

    For each stratum (block), conditions on the observed number of cases,
    eliminating the intercept.  Equivalent to the Cox partial likelihood
    within each matched set.  Concordant blocks (all cases or all controls)
    contribute 0.  Handles blocks of size 2 and 3.

    Assumes blocks are contiguous in G and y and sorted by size descending
    (as returned by reduce_to_relatives).

    Parameters
    ----------
    G : array, shape (M,)
        Genotypic values (linear predictor, no intercept).
    y : array, shape (M,)
        Binary outcomes.
    block_sizes : list of int
        Sizes of contiguous blocks.

    Returns
    -------
    float : total conditional log-likelihood.
    """
    from collections import Counter
    size_counts = Counter(block_sizes)
    log_lik = jnp.zeros(())
    offset = 0

    for sz in sorted(size_counts.keys(), reverse=True):
        n = size_counts[sz]
        end = offset + n * sz
        G_g = G[offset:end].reshape(n, sz)
        y_g = y[offset:end].reshape(n, sz)
        m = y_g.sum(axis=1)
        num = (G_g * y_g).sum(axis=1)

        if sz == 2:
            den = jax.scipy.special.logsumexp(G_g, axis=1)
            cll = jnp.where(m == 1, num - den, 0.0)
        elif sz == 3:
            den1 = jax.scipy.special.logsumexp(G_g, axis=1)
            pairs = jnp.stack([
                G_g[:, 0] + G_g[:, 1],
                G_g[:, 0] + G_g[:, 2],
                G_g[:, 1] + G_g[:, 2],
            ], axis=1)
            den2 = jax.scipy.special.logsumexp(pairs, axis=1)
            cll = jnp.where(m == 1, num - den1,
                   jnp.where(m == 2, num - den2, 0.0))
        else:
            raise NotImplementedError(f"Block size {sz} not supported")

        log_lik = log_lik + cll.sum()
        offset = end

    return log_lik


def lr_conditional(M=None, L=None, y=None, block_sizes=None,
                   muprior=dist.HalfCauchy, priorscale=0.5):
    """Logistic regression with conditional likelihood on blocked strata.

    Gaussian random effects combined with the conditional logistic likelihood
    that conditions on the number of cases per block, eliminating the intercept.
    Equivalent to conditional logistic / Cox partial likelihood within matched
    sets.  Concordant strata contribute 0 log-likelihood.

    No discrete latent variables: sample with plain NUTS.

    Parameters
    ----------
    M : int
        Number of individuals in the relatives-only reduced sample.
    L : array, shape (M, M)
        Cholesky factor of the genetic relationship matrix.
    y : array, shape (M,)
        Binary outcomes (case=1, control=0).
    block_sizes : list of int
        Sizes of contiguous family blocks in L and y.
    muprior : distribution class or instance
        Prior for mu.  If a class, called as muprior(priorscale).
    priorscale : float
        Scale parameter passed to muprior when muprior is a class.
    """
    assert block_sizes is not None, "block_sizes must be provided"
    mu_dist = muprior if isinstance(muprior, dist.Distribution) else muprior(priorscale)
    mu = numpyro.sample("mu", mu_dist)
    s = numpyro.deterministic("s", jnp.sqrt(2.0 * mu))

    with numpyro.plate("M individuals", M):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))

    G = s * jnp.matmul(L, z)
    numpyro.factor("cond_loglik", cond_loglik_blocks(G, y, block_sizes))


# ---------------------------------------------------------------------
# Ascertainment-corrected likelihood for sibships with >= 1 case
# ---------------------------------------------------------------------
def _asc_loglik_group(G_blocks, y_blocks, beta0):
    """Ascertainment-corrected log-likelihood for a batch of same-size blocks.

    Each block was selected because it contains at least one case.
    log L = log P(y|G) - log P(at least one case|G)
          = [beta0*m + sum_cases G_i] - log[prod_i(1+exp(beta0+G_i)) - 1]
          = log_num - log(exp(S) - 1),  S = sum_i softplus(beta0+G_i)

    Parameters
    ----------
    G_blocks : array (n_blocks, block_size)
    y_blocks : array (n_blocks, block_size)
    beta0    : scalar

    Returns
    -------
    Scalar: sum of per-block log-likelihoods.
    """
    log_num = beta0 * y_blocks.sum(axis=1) + (G_blocks * y_blocks).sum(axis=1)
    S = jax.nn.softplus(beta0 + G_blocks).sum(axis=1)
    log_denom = jnp.where(S > 30.0, S,
                          jnp.log(jnp.expm1(jnp.clip(S, 1e-30, 30.0))))
    return (log_num - log_denom).sum()


def lr_asc(
    L_blocks_3=None,
    L_blocks_2=None,
    y_flat=None,
    n3=0,
    n2=0,
    p_obs=None,
    muprior=dist.HalfCauchy,
    priorscale=0.5,
):
    """lr_discrete extended with ascertainment-corrected likelihood.

    For sibships ascertained by having at least one affected member.
    Handles size-3 (full-sib triplet) and size-2 (pair) blocks; size-3
    blocks must come first in y_flat.  Use with DiscreteHMCGibbs.

    Parameters
    ----------
    L_blocks_3 : array (n3, 3, 3), Cholesky factors for size-3 blocks
    L_blocks_2 : array (n2, 2, 2), Cholesky factors for size-2 blocks
    y_flat     : array (3*n3 + 2*n2,), outcomes block-contiguous (size-3 first)
    n3, n2     : int, number of size-3 and size-2 blocks
    p_obs      : float, population prevalence for Beta prior on p
    muprior    : distribution class or instance for mu
    priorscale : float, scale for muprior when a class
    """
    M = 3 * n3 + 2 * n2
    assert p_obs is not None, "p_obs (population prevalence) must be provided"

    K_conc = 20.0
    p = numpyro.sample("p", dist.Beta(K_conc * p_obs, K_conc * (1.0 - p_obs)))
    beta0 = jnp.log(p / (1.0 - p))

    mu_dist = muprior if isinstance(muprior, dist.Distribution) else muprior(priorscale)
    mu = numpyro.sample("mu", mu_dist)
    s = numpyro.deterministic("s", jnp.sqrt(2.0 * mu))

    component_loc = jnp.stack([-mu, mu])
    component_scale = jnp.stack([s, s])
    mixing_probs = jnp.array([1.0 - p, p])

    with numpyro.plate("M individuals", M, dim=-1):
        z = numpyro.sample("z", dist.Categorical(probs=mixing_probs),
                           infer={"enumerate": "sequential"})
        loc_z = Vindex(component_loc)[z]
        scale_z = Vindex(component_scale)[z]
        with handlers.reparam(config={"Z_mix": LocScaleReparam(centered=0)}):
            Z_mix = numpyro.sample("Z_mix", dist.Normal(loc=loc_z, scale=scale_z))

    Z = Z_mix - (2.0 * p - 1.0) * mu
    log_lik = jnp.zeros(())

    if n3 > 0:
        Z3 = Z[:3 * n3].reshape(n3, 3)
        G3 = jnp.einsum('bi,bji->bj', Z3, L_blocks_3)
        y3 = y_flat[:3 * n3].reshape(n3, 3)
        log_lik = log_lik + _asc_loglik_group(G3, y3, beta0)

    if n2 > 0:
        Z2 = Z[3 * n3:].reshape(n2, 2)
        G2 = jnp.einsum('bi,bji->bj', Z2, L_blocks_2)
        y2 = y_flat[3 * n3:].reshape(n2, 2)
        log_lik = log_lik + _asc_loglik_group(G2, y2, beta0)

    numpyro.factor("ascertained_loglik", log_lik)


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
    n_fullsib_trips=50_000,
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
    full-sib pairs (correlation 0.5), full-sib triplets (correlation 0.5),
    half-sib pairs (correlation 0.25), and unrelated individuals.  Disease
    status follows a logistic model P(y=1|g) = expit(alpha + g), with alpha
    calibrated to target prevalence K.

    The case-control sample includes all cases and an equal number of randomly
    sampled controls.  The returned relationship matrix and its Cholesky factor
    correspond to the sampled individuals only.

    Triplet model: two unrelated parents with genotypic values G_i, G_j ~
    N(0, sigma^2).  Each offspring k draws g_i ~ N(0, sigma^2) with
    cor(g_i, G_i) = 0.5 and g_j ~ N(0, sigma^2) with cor(g_j, G_j) = 0.5,
    using the standard infinitesimal model G_k = (G_i + G_j)/2 + m_k where
    m_k ~ N(0, sigma^2/2) independent across offspring.  This gives
    Var(G_k) = sigma^2 and pairwise sibling correlation 0.5.

    Note: Under the logistic link, lambda_S < exp(mu) because the logistic
    function saturates for high-g individuals who dominate E[risk1*risk2].
    The approximation lambda_S ≈ exp(mu) is good for mu < ~1 (ratio > 0.9)
    but deteriorates as mu increases.

    Parameters
    ----------
    n_fullsib_pairs : int
        Number of full-sib pairs (genetic correlation 0.5).
    n_fullsib_trips : int
        Number of full-sib triplets (3 siblings per family, pairwise
        genetic correlation 0.5).
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
    # Layout: [fs_pairs | fs_trips | hs_pairs | unrel]
    N = 2 * n_fullsib_pairs + 3 * n_fullsib_trips + 2 * n_halfsib_pairs + n_unrelated
    trip_offset = 2 * n_fullsib_pairs
    hs_offset = 2 * n_fullsib_pairs + 3 * n_fullsib_trips

    # --- Generate genotypic values per family type ---
    # Full-sib pairs: bivariate normal, correlation 0.5
    z1 = rng.standard_normal(n_fullsib_pairs)
    z2 = rng.standard_normal(n_fullsib_pairs)
    g_fs1 = sigma * z1
    g_fs2 = sigma * (0.5 * z1 + np.sqrt(0.75) * z2)

    # Full-sib triplets: standard infinitesimal model
    # G_k = (G_i + G_j)/2 + m_k,  m_k ~ N(0, sigma^2/2) independent per offspring
    # cor(G_k, G_i) = 0.5 (parent-offspring);  cor(G_k, G_l) = 0.5 (full-sib)
    G_par1 = sigma * rng.standard_normal(n_fullsib_trips)
    G_par2 = sigma * rng.standard_normal(n_fullsib_trips)
    midpar = (G_par1 + G_par2) / 2                                   # N(0, sigma^2/2)
    mend = (sigma / np.sqrt(2)) * rng.standard_normal((n_fullsib_trips, 3))  # N(0, sigma^2/2)
    g_trips_mat = midpar[:, None] + mend   # shape (n_fullsib_trips, 3), each col ~ N(0, sigma^2)
    # Verify pairwise sibling correlations (should be ~0.5 for all three pairs)
    trip_sib_cors = [
        float(np.corrcoef(g_trips_mat[:, a], g_trips_mat[:, b])[0, 1])
        for a, b in [(0, 1), (0, 2), (1, 2)]
    ]
    g_trips = g_trips_mat.ravel()  # interleaved: [fam0_sib0, fam0_sib1, fam0_sib2, fam1_sib0, ...]

    # Half-sib pairs: bivariate normal, correlation 0.25
    z1 = rng.standard_normal(n_halfsib_pairs)
    z2 = rng.standard_normal(n_halfsib_pairs)
    g_hs1 = sigma * z1
    g_hs2 = sigma * (0.25 * z1 + np.sqrt(1 - 0.0625) * z2)

    # Unrelated individuals
    g_unrel = sigma * rng.standard_normal(n_unrelated)

    g = np.concatenate([
        np.column_stack([g_fs1, g_fs2]).ravel(),
        g_trips,
        np.column_stack([g_hs1, g_hs2]).ravel(),
        g_unrel,
    ])

    # --- Find intercept alpha for target prevalence K ---
    alpha = brentq(lambda a: np.mean(expit(a + g)) - K, -30, 10)

    # --- Simulate disease status ---
    y = rng.binomial(1, expit(alpha + g)).astype(int)
    observed_K = y.mean()

    # --- Empirical lambda_S from all full-sib pairs (pairs + triplets) ---
    y_fs = y[:2 * n_fullsib_pairs].reshape(n_fullsib_pairs, 2)
    y_trips_pop = y[trip_offset: trip_offset + 3 * n_fullsib_trips].reshape(n_fullsib_trips, 3)

    n_probands = 0
    n_recurrences = 0
    # From full-sib pairs
    for a, b in [(0, 1), (1, 0)]:
        mask = y_fs[:, a] == 1
        n_probands += int(mask.sum())
        n_recurrences += int(y_fs[mask, b].sum())
    # From triplet families (3 pairs per family)
    for a, b in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]:
        mask = y_trips_pop[:, a] == 1
        n_probands += int(mask.sum())
        n_recurrences += int(y_trips_pop[mask, b].sum())

    empirical_lambda_S = (n_recurrences / n_probands / observed_K) if n_probands > 0 else np.nan

    n_concordant_case_fs = int((y_fs.sum(axis=1) == 2).sum())
    n_concordant_case_trips = int((y_trips_pop.sum(axis=1) == 3).sum())

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

    # Full-sib pairs: original indices (2k, 2k+1)
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

    # Full-sib triplets: 3 pairs per family — (0,1), (0,2), (1,2)
    n_sampled_trip_pairs = 0
    n_sampled_aff_trip_pairs = 0
    n_sampled_disc_trip_pairs = 0
    n_sampled_conc_ctrl_trip_pairs = 0
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        t_first = trip_offset + np.arange(n_fullsib_trips) * 3 + a
        t_second = trip_offset + np.arange(n_fullsib_trips) * 3 + b
        p1 = np.clip(np.searchsorted(sample_idx, t_first), 0, M - 1)
        p2 = np.clip(np.searchsorted(sample_idx, t_second), 0, M - 1)
        both_in = (sample_idx[p1] == t_first) & (sample_idx[p2] == t_second)
        A_sample[p1[both_in], p2[both_in]] = 0.5
        A_sample[p2[both_in], p1[both_in]] = 0.5
        ya = y_trips_pop[:, a]
        yb = y_trips_pop[:, b]
        n_sampled_trip_pairs += int(both_in.sum())
        n_sampled_aff_trip_pairs += int(np.sum(both_in & (ya == 1) & (yb == 1)))
        n_sampled_disc_trip_pairs += int(np.sum(both_in & (ya != yb)))
        n_sampled_conc_ctrl_trip_pairs += int(np.sum(both_in & (ya == 0) & (yb == 0)))

    # Half-sib pairs
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
        n_fullsib_trips=n_fullsib_trips,
        n_halfsib_pairs=n_halfsib_pairs,
        n_unrelated=n_unrelated,
        n_cases_pop=n_cases,
        n_concordant_case_fullsib_pop=n_concordant_case_fs,
        n_concordant_case_trips_pop=n_concordant_case_trips,
        M=M,
        n_cases_sample=int(y_sample.sum()),
        n_controls_sample=int(M - y_sample.sum()),
        n_sampled_fullsib_pairs=n_sampled_fs_pairs,
        n_sampled_aff_fullsib_pairs=n_sampled_aff_fs_pairs,
        n_sampled_trip_pairs=n_sampled_trip_pairs,
        n_sampled_aff_trip_pairs=n_sampled_aff_trip_pairs,
        n_sampled_disc_trip_pairs=n_sampled_disc_trip_pairs,
        n_sampled_conc_ctrl_trip_pairs=n_sampled_conc_ctrl_trip_pairs,
        n_sampled_halfsib_pairs=n_sampled_hs_pairs,
        theoretical_lambda_S=theoretical_lambda_S,
        empirical_lambda_S=empirical_lambda_S,
        trip_sib_cors=trip_sib_cors,
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
    print(f"    Full-sib triplets:   {info.get('n_fullsib_trips', 0):>12,}")
    print(f"    Half-sib pairs:      {info['n_halfsib_pairs']:>12,}")
    print(f"    Unrelated:           {info['n_unrelated']:>12,}")
    print(f"  Cases in population:   {info['n_cases_pop']:>12,}")
    print(f"  Observed prevalence:   {info['K_observed']:.6f}"
          f"  (target: {info['K_target']:.4f})")
    print(f"  Concordant-affected full-sib pairs:    {info['n_concordant_case_fullsib_pop']}")
    print(f"  Concordant-affected full-sib triplets: {info.get('n_concordant_case_trips_pop', 'n/a')}")

    if 'trip_sib_cors' in info:
        cors = info['trip_sib_cors']
        print(f"  Triplet pairwise sib correlations: "
              f"{cors[0]:.4f}, {cors[1]:.4f}, {cors[2]:.4f}  (expected 0.5)")

    print("\n=== Sibling recurrence risk ratio (lambda_S) ===")
    print(f"  Theoretical exp(mu):         {info['theoretical_lambda_S']:.4f}")
    print(f"  Empirical (all full sibs):   {info['empirical_lambda_S']:.4f}")
    ratio = info['empirical_lambda_S'] / info['theoretical_lambda_S']
    print(f"  Ratio empirical/theoretical: {ratio:.4f}")

    print("\n=== Case-control sample ===")
    print(f"  Sample size M:         {info['M']:>6}")
    print(f"    Cases:               {info['n_cases_sample']:>6}")
    print(f"    Controls:            {info['n_controls_sample']:>6}")
    print(f"  Full-sib pairs (both sampled):          {info['n_sampled_fullsib_pairs']}")
    print(f"    of which concordant-affected:         {info['n_sampled_aff_fullsib_pairs']}")
    if 'n_sampled_trip_pairs' in info:
        n = info['n_sampled_trip_pairs']
        ca = info['n_sampled_aff_trip_pairs']
        cu = info['n_sampled_conc_ctrl_trip_pairs']
        d = info['n_sampled_disc_trip_pairs']
        print(f"  Triplet sib-pairs (both sampled):       {n}")
        print(f"    concordant-affected (case-case):      {ca}")
        print(f"    concordant-unaffected (ctrl-ctrl):    {cu}")
        print(f"    discordant:                           {d}")
    print(f"  Half-sib pairs (both sampled):          {info['n_sampled_halfsib_pairs']}")

    print("\n=== True parameters ===")
    print(f"  mu       = {info['mu']:.4f}  (expected log-LR, nats)")
    print(f"  s        = {info['s']:.4f}")
    print(f"  alpha    = {info['alpha']:.4f}  (intercept)")
    print(f"  K        = {info['K_target']:.4f}")


# ---------------------------------------------------------------------
# Ascertained-sibship simulation
# ---------------------------------------------------------------------
def simulate_ascertained_sibships(
    n_fullsib_pairs=100_000,
    n_fullsib_trips=50_000,
    n_halfsib_pairs=10_000,
    K=0.01,
    mu=1.0,
    seed=42,
):
    """Simulate population and select all members of sibships with >= 1 case.

    Generates the same population as simulate_casecontrol_related but uses
    a different ascertainment rule: a sibship (pair or triplet) is included
    in full whenever at least one member is affected.  No singletons or
    unrelated individuals are included.

    The returned arrays are block-contiguous with size-3 blocks (full-sib
    triplets) first, then size-2 blocks (full-sib pairs, then half-sib
    pairs).  Per-block Cholesky factors are returned as two arrays,
    L_blocks_3 and L_blocks_2, tiled from the block-type prototype.

    Parameters
    ----------
    n_fullsib_pairs, n_fullsib_trips, n_halfsib_pairs : int
        Population counts of each relationship type.
    K : float
        Target population prevalence.
    mu : float
        Genetic information parameter (nats); marginal variance = 2*mu.
    seed : int

    Returns
    -------
    y_sample   : ndarray (M,), binary outcomes
    L_blocks_3 : ndarray (n3, 3, 3), Cholesky factors for size-3 blocks
    L_blocks_2 : ndarray (n2, 2, 2), Cholesky factors for size-2 blocks
    n3, n2     : int, number of size-3 and size-2 blocks
    info       : dict, simulation parameters and diagnostics
    """
    from scipy.optimize import brentq
    from scipy.special import expit
    from collections import Counter

    rng = np.random.default_rng(seed)
    sigma = np.sqrt(2 * mu)

    N = 2 * n_fullsib_pairs + 3 * n_fullsib_trips + 2 * n_halfsib_pairs
    trip_offset = 2 * n_fullsib_pairs
    hs_offset   = trip_offset + 3 * n_fullsib_trips

    # --- Generate genotypic values (identical to simulate_casecontrol_related) ---
    z1 = rng.standard_normal(n_fullsib_pairs)
    z2 = rng.standard_normal(n_fullsib_pairs)
    g_fs1 = sigma * z1
    g_fs2 = sigma * (0.5 * z1 + np.sqrt(0.75) * z2)

    G_par1 = sigma * rng.standard_normal(n_fullsib_trips)
    G_par2 = sigma * rng.standard_normal(n_fullsib_trips)
    midpar = (G_par1 + G_par2) / 2
    mend   = (sigma / np.sqrt(2)) * rng.standard_normal((n_fullsib_trips, 3))
    g_trips_mat = midpar[:, None] + mend           # (n_fullsib_trips, 3)

    z1 = rng.standard_normal(n_halfsib_pairs)
    z2 = rng.standard_normal(n_halfsib_pairs)
    g_hs1 = sigma * z1
    g_hs2 = sigma * (0.25 * z1 + np.sqrt(1 - 0.0625) * z2)

    g = np.concatenate([
        np.column_stack([g_fs1, g_fs2]).ravel(),
        g_trips_mat.ravel(),
        np.column_stack([g_hs1, g_hs2]).ravel(),
    ])

    alpha = brentq(lambda a: np.mean(expit(a + g)) - K, -30, 10)
    y = rng.binomial(1, expit(alpha + g)).astype(int)
    K_obs = y.mean()

    # --- Empirical lambda_S ---
    y_fs_pop    = y[:2 * n_fullsib_pairs].reshape(n_fullsib_pairs, 2)
    y_trip_pop  = y[trip_offset:trip_offset + 3 * n_fullsib_trips].reshape(n_fullsib_trips, 3)
    y_hs_pop    = y[hs_offset:hs_offset + 2 * n_halfsib_pairs].reshape(n_halfsib_pairs, 2)

    n_prob, n_rec = 0, 0
    for mat, pairs in [
        (y_fs_pop,   [(0,1),(1,0)]),
        (y_trip_pop, [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]),
    ]:
        for a, b in pairs:
            mask = mat[:, a] == 1
            n_prob += int(mask.sum())
            n_rec  += int(mat[mask, b].sum())
    empirical_lambda_S = (n_rec / n_prob / K_obs) if n_prob > 0 else np.nan

    # --- Ascertain sibships with >= 1 case ---
    asc_trips = y_trip_pop.sum(axis=1) >= 1
    asc_fs    = y_fs_pop.sum(axis=1)   >= 1
    asc_hs    = y_hs_pop.sum(axis=1)   >= 1

    n_asc_trips = int(asc_trips.sum())
    n_asc_fs    = int(asc_fs.sum())
    n_asc_hs    = int(asc_hs.sum())

    # Outcomes in block-contiguous order: size-3 first, then size-2
    y_flat = np.concatenate([
        y_trip_pop[asc_trips].ravel(),   # (n_asc_trips * 3,)
        y_fs_pop[asc_fs].ravel(),        # (n_asc_fs   * 2,)
        y_hs_pop[asc_hs].ravel(),        # (n_asc_hs   * 2,)
    ])

    n3 = n_asc_trips
    n2 = n_asc_fs + n_asc_hs
    M  = 3 * n3 + 2 * n2

    # --- Prototype Cholesky factors ---
    L_trip = np.linalg.cholesky(np.array([[1., .5, .5],[.5, 1., .5],[.5, .5, 1.]]))
    L_fs   = np.linalg.cholesky(np.array([[1., .5 ],[.5,  1.]]))
    L_hs   = np.linalg.cholesky(np.array([[1., .25],[.25, 1.]]))

    L_blocks_3 = np.tile(L_trip[None], (n3, 1, 1)) if n3 > 0 else np.zeros((0, 3, 3))
    L_blocks_2 = np.concatenate([
        np.tile(L_fs[None], (n_asc_fs, 1, 1)) if n_asc_fs > 0 else np.zeros((0, 2, 2)),
        np.tile(L_hs[None], (n_asc_hs, 1, 1)) if n_asc_hs > 0 else np.zeros((0, 2, 2)),
    ], axis=0) if n2 > 0 else np.zeros((0, 2, 2))

    # --- Block composition: cases per block ---
    block3_comp = Counter(y_flat[:3*n3].reshape(n3, 3).sum(axis=1).astype(int).tolist()) if n3 > 0 else {}
    block2_comp = Counter(y_flat[3*n3:].reshape(n2, 2).sum(axis=1).astype(int).tolist()) if n2 > 0 else {}

    info = dict(
        N=N, K_target=K, K_observed=K_obs, alpha=alpha, mu=mu, s=sigma,
        n_fullsib_pairs=n_fullsib_pairs, n_fullsib_trips=n_fullsib_trips,
        n_halfsib_pairs=n_halfsib_pairs,
        n_asc_trips=n_asc_trips, n_asc_fs=n_asc_fs, n_asc_hs=n_asc_hs,
        n3=n3, n2=n2, M=M,
        n_cases=int(y_flat.sum()),
        block3_comp=dict(block3_comp),
        block2_comp=dict(block2_comp),
        theoretical_lambda_S=np.exp(mu),
        empirical_lambda_S=empirical_lambda_S,
    )
    return y_flat, L_blocks_3, L_blocks_2, n3, n2, info


def print_ascertained_summary(info):
    """Print a summary of an ascertained-sibship simulation."""
    print("=== Population ===")
    print(f"  Total individuals:      {info['N']:>10,}")
    print(f"    Full-sib pairs:       {info['n_fullsib_pairs']:>10,}")
    print(f"    Full-sib triplets:    {info['n_fullsib_trips']:>10,}")
    print(f"    Half-sib pairs:       {info['n_halfsib_pairs']:>10,}")
    print(f"  Observed prevalence:    {info['K_observed']:.6f}  (target: {info['K_target']:.4f})")
    print(f"\n  lambda_S  theoretical: {info['theoretical_lambda_S']:.4f}"
          f"   empirical: {info['empirical_lambda_S']:.4f}")

    print(f"\n=== Ascertained sibships (>= 1 affected member) ===")
    print(f"  Triplet blocks (size 3): {info['n_asc_trips']:>5}")
    for m, cnt in sorted(info.get('block3_comp', {}).items()):
        print(f"    {m} case(s), {3-m} control(s): {cnt:>5}")
    print(f"  Pair blocks (size 2):    {info['n2']:>5}"
          f"  ({info['n_asc_fs']} full-sib, {info['n_asc_hs']} half-sib)")
    for m, cnt in sorted(info.get('block2_comp', {}).items()):
        print(f"    {m} case(s), {2-m} control(s): {cnt:>5}")
    print(f"\n  Total M: {info['M']}   cases: {info['n_cases']}"
          f"   controls: {info['M'] - info['n_cases']}")

    print(f"\n=== True parameters ===")
    print(f"  mu = {info['mu']:.4f}   s = {info['s']:.4f}   alpha = {info['alpha']:.4f}")


def create_data(seed, K, lambdaS):
    """Simple proband-sibling pair simulator used by polyagamma_gibbs.py.

    Returns (M, L, y, y, empirical_log_lambdaS).
    """
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.columns import Columns

    SEED = seed
    N_PAIRS = 80_000
    POP_K = K
    SIB_K = lambdaS * K
    rng = np.random.default_rng(SEED)

    y1 = rng.random(N_PAIRS) < POP_K
    y2 = rng.random(N_PAIRS) < np.where(y1, SIB_K, POP_K)
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
    table.add_row("Sibling control", f"{((y1==0)&(y2==0)).sum()}", f"{((y1==1)&(y2==0)).sum()}", f"{(y2==0).sum()}")
    table.add_row("Sibling case",    f"{((y1==0)&(y2==1)).sum()}", f"{((y1==1)&(y2==1)).sum()}", f"{(y2==1).sum()}")
    table.add_row("Total",           f"{(y1==0).sum()}",           f"{(y1==1).sum()}",           f"{len(y1)}")
    table2 = table
    emplogs = np.log(y2[y1==1].mean() / y1.mean())

    y_all      = np.concatenate([y1, y2]).astype(int)
    fam_id_all = np.concatenate([np.arange(N_PAIRS), np.arange(N_PAIRS)])

    case_idx      = np.where(y_all == 1)[0]
    n_cases       = len(case_idx)
    ctrl_idx_pool = np.where(y_all == 0)[0]
    ctrl_idx      = rng.choice(ctrl_idx_pool, size=n_cases, replace=False)
    keep_idx      = np.sort(np.concatenate([case_idx, ctrl_idx]))
    y             = y_all[keep_idx]
    fam_id        = fam_id_all[keep_idx]
    M             = len(keep_idx)

    G = np.eye(M, dtype=float)
    y1_sel, y2_sel = [], []
    for fid in np.unique(fam_id):
        members = np.where(fam_id == fid)[0]
        if len(members) == 2:
            i, j = members
            G[i, j] = G[j, i] = 0.5
            y1_sel.append(y[i])
            y2_sel.append(y[j])
    y1_sel = np.array(y1_sel)
    y2_sel = np.array(y2_sel)

    table = Table(title=f"Case-control sample  M={M}  ({n_cases} cases, {n_cases} controls)")
    table.add_column("")
    table.add_column("Proband control")
    table.add_column("Proband case")
    table.add_column("Total")
    table.add_row("Sibling control", f"{((y1_sel==0)&(y2_sel==0)).sum()}", f"{((y1_sel==1)&(y2_sel==0)).sum()}", f"{(y2_sel==0).sum()}")
    table.add_row("Sibling case",    f"{((y1_sel==0)&(y2_sel==1)).sum()}", f"{((y1_sel==1)&(y2_sel==1)).sum()}", f"{(y2_sel==1).sum()}")
    table.add_row("Total",           f"{(y1_sel==0).sum()}",               f"{(y1_sel==1).sum()}",               f"{len(y1_sel)}")
    table3 = table
    console.print(Panel(Group(Columns([table1, table2, table3], expand=True)), title="Data simulation"))

    L = np.linalg.cholesky(G)
    return M, L, y, y, emplogs


# ---------------------------------------------------------------------
# Main: quick simulation test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    y, A, L, info = simulate_casecontrol_related(
        n_fullsib_pairs=100_000, n_fullsib_trips=50_000,
        n_halfsib_pairs=10_000, n_unrelated=1000, mu=1.0,
    )
    print_casecontrol_summary(info)
