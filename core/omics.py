"""High-dimensional hypothesis-testing design-stage simulation.

Important: Monte Carlo draws *asymptotic z test statistics*, not RNA-seq reads.
The NB mean/dispersion motivates a delta-method standard error. Test statistics
are assumed marginally N(delta, 1), with independent features. This is NOT
DESeq2, edgeR, muscat, scPower, or a gene-level NB GLM implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class OmicsConfig:
    assay: str = 'bulk_rnaseq'  # bulk_rnaseq | standardized
    features: int = 4000
    pi1: float = 0.05
    fold_change: float = 1.5
    median_count: float = 50.0
    dispersion: float = 0.20
    log_count_sd: float = 0.6
    log_dispersion_sd: float = 0.35
    standardized_effect: float = 0.65
    fdr: float = 0.05
    target_power: float = 0.80
    repetitions: int = 100
    seed: int = 20261027


def validate_config(config: OmicsConfig) -> None:
    if config.assay not in ('bulk_rnaseq', 'standardized'):
        raise ValueError('Unsupported assay.')
    if not 100 <= config.features <= 25000:
        raise ValueError('Features must be between 100 and 25,000.')
    if not 0 < config.pi1 < 1:
        raise ValueError('Non-null feature fraction must be in (0,1).')
    if not 0 < config.fdr < 1 or not 0 < config.target_power < 1:
        raise ValueError('FDR and target power must be between 0 and 1.')
    if config.repetitions < 10 or config.repetitions > 2000:
        raise ValueError('Use 10 to 2,000 Monte Carlo repetitions.')
    if config.assay == 'bulk_rnaseq':
        if config.fold_change <= 1 or config.median_count <= 0 or config.dispersion <= 0:
            raise ValueError('Fold change >1, mean count >0 and dispersion >0 required.')
    elif config.standardized_effect <= 0:
        raise ValueError('Standardized effect must be positive.')
    if config.log_count_sd < 0 or config.log_dispersion_sd < 0:
        raise ValueError('Log SD must not be negative.')


def profile_from_csv(upload: bytes, assay: str) -> pd.DataFrame:
    """Return feature-wise inputs from a small, non-identifying CSV.
    RNA requires mean_count,dispersion; standardized requires feature_sd (optional
    optional feature_sd is not yet used in current standardized-effect model).
    """
    from io import BytesIO
    if len(upload) > 5_000_000:
        raise ValueError('CSV exceeds 5 MB limit.')
    try:
        frame = pd.read_csv(BytesIO(upload))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise ValueError('CSV is empty, corrupted or not valid UTF-8 tabular text.') from exc
    if len(frame) < 100 or len(frame) > 25000:
        raise ValueError('CSV must have 100-25,000 feature rows.')
    if assay == 'bulk_rnaseq':
        required = {'mean_count', 'dispersion'}
        if not required.issubset(frame.columns):
            raise ValueError('RNA-seq CSV requires mean_count and dispersion columns.')
        data = frame[['mean_count','dispersion']].apply(pd.to_numeric, errors='raise')
        if not np.isfinite(data.to_numpy()).all() or (data <= 0).any().any():
            raise ValueError('RNA-seq means and dispersions must be finite and positive.')
        return data.reset_index(drop=True)
    raise ValueError('Uploaded profile currently supported for RNA-seq only.')


def _bh_rejections(p: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg rejection mask for a vector of p-values."""
    m = p.size
    order = np.argsort(p)
    sorted_p = p[order]
    passing = sorted_p <= alpha * np.arange(1, m+1) / m
    mask = np.zeros(m, dtype=bool)
    if np.any(passing):
        cutoff = int(np.flatnonzero(passing)[-1])
        mask[order[:cutoff+1]] = True
    return mask


def run_omics_curve(config: OmicsConfig, n_values: list[int],
                    pilot: pd.DataFrame | None = None) -> pd.DataFrame:
    """FDR-adjusted average gene-level discovery power under z-test approximation.

    Monte Carlo uncertainty CIs are across repeated synthetic studies *conditional
    on assumptions*, not confidence bounds on biologically unknown parameters.
    """
    validate_config(config)
    ns = sorted(set(int(n) for n in n_values))
    if not ns or ns[0] < 3 or ns[-1] > 2000 or len(ns) > 30:
        raise ValueError('Sample-size grid: 3-2000 per group, max 30 distinct values.')
    if config.features * config.repetitions * len(ns) > 25_000_000 and pilot is None:
        raise ValueError('Computational budget exceeded: features × repetitions × grid size must be ≤25,000,000. Reduce repetitions or feature count.')
    if pilot is not None and len(pilot) * config.repetitions * len(ns) > 25_000_000:
        raise ValueError('Computational budget exceeded for pilot profile: profile rows × repetitions × grid size must be ≤25,000,000.')
    rng = np.random.default_rng(config.seed)
    if pilot is not None:
        if config.assay != 'bulk_rnaseq':
            raise ValueError('Pilot profiles only supported for RNA-seq.')
        means = pilot['mean_count'].to_numpy(dtype=float)
        dispersions = pilot['dispersion'].to_numpy(dtype=float)
        m = len(means)
    else:
        m = config.features
        if config.assay == 'bulk_rnaseq':
            means = rng.lognormal(mean=np.log(config.median_count),
                                  sigma=config.log_count_sd, size=m)
            dispersions = rng.lognormal(mean=np.log(config.dispersion),
                                        sigma=config.log_dispersion_sd, size=m)
    n_nonnull = max(1, min(m - 1, int(round(m * config.pi1))))
    rows = []
    for n in ns:
        if config.assay == 'bulk_rnaseq':
            # Signal has a log-ratio effect. Var(log group mean) approximately
            # (1/mu+phi)/n for an NB count and equal-sized groups.
            # For conservative planning all target changes are treated as upregulated.
            varlog = ((1 / means + dispersions) +
                      (1 / (means * config.fold_change) + dispersions)) / n
            noncentrality = np.log(config.fold_change) / np.sqrt(varlog)
        else:
            noncentrality = np.full(m, config.standardized_effect * np.sqrt(n / 2))
        power_records, fdp_records, discoveries, false_pos = [], [], [], []
        for _ in range(config.repetitions):
            alt_idx = rng.choice(m, n_nonnull, replace=False)
            truth = np.zeros(m, dtype=bool)
            truth[alt_idx] = True
            z = rng.standard_normal(m)
            z[truth] += noncentrality[truth]
            p = np.minimum(1.0, 2.0 * norm.sf(np.abs(z)))
            rejected = _bh_rejections(p, config.fdr)
            tp = int(np.count_nonzero(rejected & truth))
            fp = int(np.count_nonzero(rejected & ~truth))
            calls = tp + fp
            power_records.append(tp / n_nonnull)
            fdp_records.append(fp / max(calls, 1))
            discoveries.append(calls)
            false_pos.append(fp)
        power_sd = float(np.std(power_records, ddof=1))
        se = power_sd / np.sqrt(config.repetitions)
        mean_power = float(np.mean(power_records))
        rows.append({'n_per_group': n, 'expected_power': mean_power,
                     'power_CI_low_MC': max(0., mean_power-1.96*se),
                     'power_CI_high_MC': min(1., mean_power+1.96*se),
                     'mean_false_discovery_proportion': float(np.mean(fdp_records)),
                     'mean_discoveries': float(np.mean(discoveries)),
                     'mean_false_positive_calls': float(np.mean(false_pos)),
                     'target_power_reached': bool(mean_power >= config.target_power)})
    return pd.DataFrame(rows)


def recommend_n(curve: pd.DataFrame) -> int | None:
    achieved = curve.loc[curve['target_power_reached'], 'n_per_group']
    return int(achieved.min()) if len(achieved) else None
