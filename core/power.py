"""Conventional inference and hierarchical experimental-design calculations.

Methods are design-stage approximations, not a replacement for a study statistician.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np
from scipy.stats import norm
from statsmodels.stats.power import FTestAnovaPower, NormalIndPower, TTestIndPower, TTestPower
from statsmodels.stats.proportion import proportion_effectsize


@dataclass(frozen=True)
class PowerResult:
    n_group_a: int
    n_group_b: int
    achieved_power: float
    effect_size: float
    method: str


def recruitment_n(n_analyzable: int, attrition: float) -> int:
    if n_analyzable < 1 or not (0 <= attrition < 1):
        raise ValueError('Require positive n and attrition in [0,1).')
    return ceil(n_analyzable / (1 - attrition))


def two_sample_t(delta: float, sd: float, alpha: float = 0.05,
                 target_power: float = 0.8, ratio: float = 1.0) -> PowerResult:
    if delta <= 0 or sd <= 0 or ratio <= 0 or not 0 < alpha < 1 or not 0 < target_power < 1:
        raise ValueError('Invalid effect, SD, allocation, alpha or target power.')
    d = abs(delta) / sd
    model = TTestIndPower()
    raw_n = model.solve_power(effect_size=d, alpha=alpha, power=target_power, ratio=ratio,
                              alternative='two-sided')
    n_a = max(2, ceil(raw_n))
    n_b = max(2, ceil(n_a * ratio))
    achieved = model.power(effect_size=d, nobs1=n_a, alpha=alpha, ratio=n_b / n_a,
                           alternative='two-sided')
    return PowerResult(n_a, n_b, float(achieved), d, 'Two-sided independent t test (pooled variance)')


def paired_t(delta: float, sd_difference: float, alpha: float = 0.05,
             target_power: float = 0.8) -> PowerResult:
    if delta <= 0 or sd_difference <= 0 or not 0 < alpha < 1 or not 0 < target_power < 1:
        raise ValueError('Invalid input.')
    d_z = abs(delta) / sd_difference
    model = TTestPower()
    n = max(2, ceil(model.solve_power(effect_size=d_z, alpha=alpha, power=target_power,
                                       alternative='two-sided')))
    achieved = model.power(effect_size=d_z, nobs=n, alpha=alpha, alternative='two-sided')
    return PowerResult(n, n, float(achieved), d_z, 'Paired two-sided t test (n = pairs)')


def two_proportions(p_a: float, p_b: float, alpha: float = 0.05,
                    target_power: float = 0.8, ratio: float = 1.0) -> PowerResult:
    if not (0 < p_a < 1 and 0 < p_b < 1 and p_a != p_b and ratio > 0):
        raise ValueError('Proportions must be distinct and strictly between 0 and 1.')
    h = abs(float(proportion_effectsize(p_a, p_b)))
    model = NormalIndPower()
    raw_n = model.solve_power(effect_size=h, alpha=alpha, power=target_power, ratio=ratio,
                              alternative='two-sided')
    n_a = max(2, ceil(raw_n))
    n_b = max(2, ceil(n_a * ratio))
    achieved = model.power(effect_size=h, nobs1=n_a, alpha=alpha,
                           ratio=n_b / n_a, alternative='two-sided')
    return PowerResult(n_a, n_b, float(achieved), h, 'Asymptotic two-proportion z test (Cohen h)')


def one_way_anova(cohen_f: float, groups: int, alpha: float = 0.05,
                  target_power: float = 0.8) -> PowerResult:
    if cohen_f <= 0 or groups < 2:
        raise ValueError('Positive Cohen f and at least two groups required.')
    model = FTestAnovaPower()
    n_total = model.solve_power(effect_size=cohen_f, alpha=alpha, power=target_power, k_groups=groups)
    per_group = max(2, ceil(n_total / groups))
    achieved = model.power(effect_size=cohen_f, nobs=per_group * groups,
                           alpha=alpha, k_groups=groups)
    return PowerResult(per_group, per_group * groups, float(achieved), cohen_f,
                       f'One-way ANOVA, group n then total N ({groups} groups)')


def sc_donor_power(delta: float, donor_sd: float, cell_sd: float,
                   total_cells_per_donor: int, cell_fraction: float,
                   alpha: float = 0.05, target_power: float = 0.8) -> dict:
    """A *one endpoint* donor-summary hierarchical normal approximation.
    Does not incorporate per-gene FDR, read depth, differential abundance, or NB counts.
    """
    if min(delta, donor_sd, cell_sd) <= 0 or total_cells_per_donor < 1 or not 0 < cell_fraction <= 1:
        raise ValueError('Invalid donor, cell or delta assumptions.')
    expected_cells = total_cells_per_donor * cell_fraction
    sigma = sqrt(donor_sd**2 + cell_sd**2 / expected_cells)
    result = two_sample_t(delta, sigma, alpha, target_power)
    return {'result': result, 'expected_target_cells': expected_cells,
            'sd_donor_summary': sigma,
            'missing_target_prob_approx': float(np.exp(-expected_cells))}


def ar1_effective_frames(frames: int, rho: float) -> float:
    """Effective independent count for AR(1) correlated frames.
    Uses exact finite-series closed form for variance of the frame mean.
    """
    if frames < 1 or not 0 <= rho < 1:
        raise ValueError('Frames must be >=1; rho in [0,1).')
    if frames == 1 or rho == 0:
        return float(frames)
    m = float(frames)
    r = float(rho)
    power_m1 = r ** (frames - 1)
    s1 = r * (1 - power_m1) / (1 - r)
    s2 = r * (1 - frames * power_m1 + (frames - 1) * r ** frames) / (1 - r)**2
    weighted_sum = m * s1 - s2
    denominator = m + 2 * weighted_sum
    return max(1.0, min(m, (m * m) / denominator))


def deeplabcut_power(delta: float, animal_sd: float, frame_sd: float,
                      frames_per_session: int, sessions: int, rho: float,
                      alpha: float = 0.05, target_power: float = 0.8,
                      tested_outcomes: int = 1) -> dict:
    if min(delta, animal_sd, frame_sd) <= 0 or sessions < 1 or tested_outcomes < 1:
        raise ValueError('Invalid study parameters.')
    effective = ar1_effective_frames(frames_per_session, rho)
    summary_sd = sqrt(animal_sd**2 + frame_sd**2 / (sessions * effective))
    effective_alpha = alpha / tested_outcomes
    result = two_sample_t(delta, summary_sd, effective_alpha, target_power)
    return {'result': result, 'effective_frames_per_session': effective,
            'summary_sd': summary_sd, 'adjusted_alpha': effective_alpha,
            'total_frame_count_per_animal': frames_per_session * sessions}


def wilson_half_width(p: float, n: int, confidence: float = 0.95) -> float:
    if not 0 < p < 1 or n < 1 or not 0 < confidence < 1:
        raise ValueError('Invalid proportion, n or confidence.')
    z = norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n
    return float(z * sqrt(p * (1 - p) / n + z**2 / (4 * n * n)) / denom)


def validation_precision(sensitivity: float, specificity: float, prevalence: float,
                         margin: float = 0.05, confidence: float = 0.95) -> dict:
    if not 0 < prevalence < 1 or not 0 < margin < 1:
        raise ValueError('Invalid prevalence or margin.')

    def solve(p: float) -> int:
        lo, hi = 1, 10000000
        if wilson_half_width(p, hi, confidence) > margin:
            raise ValueError('Requested precision exceeds search range.')
        while lo < hi:
            mid = (lo + hi) // 2
            if wilson_half_width(p, mid, confidence) <= margin:
                hi = mid
            else:
                lo = mid + 1
        return lo

    n_pos = solve(sensitivity)
    n_neg = solve(specificity)
    n_total = max(ceil(n_pos / prevalence), ceil(n_neg / (1 - prevalence)))
    return {'positive_cases': n_pos, 'negative_cases': n_neg,
            'total_recruitment': n_total,
            'expected_positive': n_total * prevalence,
            'expected_negative': n_total * (1-prevalence)}


def power_curve_t(effect_size: float, alpha: float, n_values: np.ndarray,
                  paired: bool = False, ratio: float = 1.0) -> np.ndarray:
    model = TTestPower() if paired else TTestIndPower()
    if paired:
        return np.asarray([model.power(effect_size, int(n), alpha=alpha) for n in n_values])
    return np.asarray([model.power(effect_size, int(n), alpha=alpha, ratio=ratio)
                       for n in n_values])
