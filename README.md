# AMRSforLS

Adaptive **M**ultiple **R**andom **S**can for **L**atent **S**pace models (AMRSforLS) — an R package with fast C++ backends for Bayesian estimation of static latent space network models under Bernoulli or Poisson likelihoods. The package implements:

- **Systematic** updates (all nodes each iteration)
- **Multiple Random Scan (MRS)** with **adaptive** selection probabilities (individual- or block-wise)
- Two adaptive proposal schemes for latent positions (Andrieu & Thoms, 2008; Łatuszyński, Roberts & Rosenthal, 2013)

---

## Why AMRSforLS?
- **Faster mixing** than plain random scan by prioritizing hard-to-move nodes/blocks.
- **Plug-and-play**: minimalist R interface, heavy lifting in C++ (Rcpp/RcppArmadillo).
- **Flexible likelihood**: Bernoulli (binary edges) or Poisson (count edges).
- **Diagnostics out-of-the-box**: per-node acceptance and time-varying selection probabilities.

---

## Installation

```r
# install.packages("remotes")
remotes::install_github("<your_github_username>/AMRSforLS")
```

> Replace `<your_github_username>` with the actual GitHub owner of this repo.

The package links to **Rcpp**, **RcppArmadillo**, and **RcppDist**.

---

## Quick start

Below are two minimal end‑to‑end examples.

### 1) Bernoulli latent space model (binary network)

```r
set.seed(123)
N  <- 30         # nodes
K  <- 2          # latent dimensions
Z0 <- matrix(rnorm(N*K, 0, 1), N, K)
D2 <- as.matrix(dist(Z0))^2
eta <- 0.5 - D2
P   <- 1/(1+exp(-eta))
Y   <- matrix(rbinom(N*N, 1, P), N, N)
Y[lower.tri(Y)] <- t(Y)[lower.tri(Y)]   # symmetrize
Y[diag(N)==1]   <- NA                    # remove self-edges

# Initial values
beta0 <- 0
z_init <- matrix(rnorm(N*K, 0, 1), N, K)

# Adaptive MRS (individual nodes), Bernoulli: distr_option = 0
out <- MCMC_ADRS_AD(
  y = Y,
  beta = beta0, beta_a = 0, beta_b = 2,
  z_i = z_init, zeta_a = 0, zeta_b = 1,
  lam_ad_z = rep(1, N),
  mu_mat_z = matrix(0, N, K),
  Sigma_ad_z = replicate(N, diag(K), simplify = FALSE),
  N = N, K = K,
  prop_sigma_beta = 0.2,
  acc_target_z = 0.234,
  scan_every = 100,
  Iterations = 2000,
  k_shift = 0,
  eq_option = 0,  # use learned selection probabilities
  eq_prob = 0.5,
  ad_option = 0,  # proposal adaptation: Andrieu & Thoms (default)
  distr_option = 0 # Bernoulli
)

str(out)
# $beta_ite : numeric vector length Iterations
# $zeta_ite : list of length Iterations, each an N x K matrix
# $P_ite    : Iterations x N matrix (per-node selection probabilities over time)
# $Acc_ite  : Iterations x N matrix (per-node acceptance rates)
```

### 2) Poisson latent space model (count network)

```r
set.seed(321)
N <- 25; K <- 2
Z0 <- matrix(rnorm(N*K, 0, 1), N, K)
D2 <- as.matrix(dist(Z0))^2
lambda <- exp(0.8 - D2)
Y <- matrix(rpois(N*N, lambda), N, N)
Y[lower.tri(Y)] <- t(Y)[lower.tri(Y)]
Y[diag(N)==1] <- NA

# Block-wise MRS: split nodes into two blocks
belong <- c(rep(1, N/2), rep(2, N - N/2))

out <- MCMC_ADRS_BLOCK_AD(
  y = Y,
  beta = 0, beta_a = 0, beta_b = 2,
  z_i = matrix(rnorm(N*K), N, K), zeta_a = 0, zeta_b = 1,
  lam_ad_z = rep(1, N), mu_mat_z = matrix(0, N, K),
  Sigma_ad_z = replicate(N, diag(K), simplify = FALSE),
  N = N, K = K, prop_sigma_beta = 0.3, acc_target_z = 0.234,
  belong = belong, scan_every = 100, Iterations = 1500,
  k_shift = 0.5, ad_option = 0, eq_option = 0,
  distr_option = 1 # Poisson
)
```

---

## Model and algorithms (high-level)
We model dyadic outcomes via a distance-based latent space:

- **Bernoulli**:  $Y_{ij} \sim \text{Bernoulli}(p_{ij})$, with $\text{logit}(p_{ij}) = \beta - \lVert z_i - z_j \rVert^2$.
- **Poisson**:   $Y_{ij} \sim \text{Poisson}(\lambda_{ij})$, with $\log \lambda_{ij} = \beta - \lVert z_i - z_j \rVert^2$.

Priors:
- $\beta \sim \mathcal{N}(\beta_a, \beta_b^2)$
- Each coordinate of $z_i$ has a normal prior $\mathcal{N}(\zeta_a, \zeta_b^2)$.

Algorithms:
- **Systematic**: update all nodes sequentially each iteration (`MCMC_AD`).
- **MRS (individual)**: update a *random subset* of nodes each iteration, with *adaptive* selection probabilities (`MCMC_ADRS_AD`).
- **MRS (blocks)**: update *random subset of blocks* of nodes; block probabilities adapt from recent acceptance (`MCMC_ADRS_BLOCK_AD`).

Adaptive proposals for \(z_i\):
- `ad_option = 0` — **Andrieu & Thoms (2008)**: per-node adaptive scale/location/covariance (`lam_ad_z`, `mu_mat_z`, `Sigma_ad_z`).
- `ad_option = 1` — **Łatuszyński, Roberts & Rosenthal (2013)**-style log-scale updates via running acceptance averages.

Selection probability adaptation (MRS):
- Every `scan_every` iterations, compute recent per-node acceptance rates and map them via a logistic rule to get updated selection probabilities.
- `eq_option = 1` forces equal probabilities (sanity check / baseline).

---

## Core functions

### `MCMC_ADRS_AD()` — Adaptive Multiple Random Scan (individual nodes)
**Usage**
```r
MCMC_ADRS_AD(y, beta, beta_a, beta_b, z_i, zeta_a, zeta_b,
             lam_ad_z, mu_mat_z, Sigma_ad_z,
             N, K,
             prop_sigma_beta,
             acc_target_z = 0.234,
             scan_every = 100,
             Iterations = 1000,
             k_shift = 0,
             eq_option = 0,
             eq_prob = 0.5,
             ad_option = 0,
             distr_option = 1)
```
**Arguments (key)**
- `y` *(N x N)*: adjacency (use `NA` on the diagonal).
- `beta` / `beta_a` / `beta_b`: intercept init / prior mean / prior sd.
- `z_i` *(N x K)*: initial latent positions.
- `zeta_a`, `zeta_b`: prior mean/sd for latent coordinates.
- `lam_ad_z`, `mu_mat_z`, `Sigma_ad_z`: A&T (2008) adaptation state; vectors/matrices/lists of size N and K.
- `prop_sigma_beta`: RW proposal sd for `beta`.
- `acc_target_z`: target acceptance for latent updates.
- `scan_every`: period (iterations) to refresh selection probabilities.
- `Iterations`: MCMC length.
- `eq_option`, `eq_prob`: force equal selection probabilities if desired.
- `ad_option`: 0=A&T(2008) proposals; 1=LRR(2013)-style.
- `distr_option`: 0=Bernoulli, 1=Poisson.

**Returns**
- `beta_ite` *(Iterations)*, `zeta_ite` *(list of N x K)*,
- `P_ite` *(Iterations x N)* selection probabilities,
- `Acc_ite` *(Iterations x N)* acceptance rates.

---

### `MCMC_ADRS_BLOCK_AD()` — Adaptive Multiple Random Scan (blocks)
**Usage**
```r
MCMC_ADRS_BLOCK_AD(y, beta, beta_a, beta_b, z_i, zeta_a, zeta_b,
                   lam_ad_z, mu_mat_z, Sigma_ad_z,
                   N, K, prop_sigma_beta, acc_target_z,
                   belong,
                   scan_every = 100, Iterations = 1000,
                   k_shift = 0.5, ad_option = 0, eq_option = 0,
                   distr_option = 1)
```
**Additional argument**
- `belong` *(length N)*: integer block label for each node. Selection probabilities are learned at the block level.

**Returns**
- Same outputs as above, with `P_ite` now *(Iterations x #blocks)*.

---

### `MCMC_AD()` — Systematic updates (no random scan)
**Usage**
```r
MCMC_AD(y, beta, beta_a, beta_b, z_i, zeta_a, zeta_b,
        lam_ad_z, mu_mat_z, Sigma_ad_z,
        N, K,
        prop_sigma_beta, acc_target_z,
        Iterations = 1000,
        ad_option = 0,
        distr_option = 1)
```
**Returns**
- `beta_ite`, `zeta_ite`, `Acc_ite`.
