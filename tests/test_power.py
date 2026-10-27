import math
import numpy as np
import pytest
from core.power import (
    ar1_effective_frames, deeplabcut_power, one_way_anova, paired_t,
    recruitment_n, sc_donor_power, two_proportions, two_sample_t,
    validation_precision, wilson_half_width,
)


def test_ttest_n_meets_target_and_effect_monotonic():
    a = two_sample_t(0.5, 1, 0.05, 0.8)
    b = two_sample_t(0.25, 1, 0.05, 0.8)
    assert 60 <= a.n_group_a <= 70
    assert a.achieved_power >= 0.8
    assert b.n_group_a > a.n_group_a


def test_paired_counts_pairs_not_measurements():
    r = paired_t(0.5, 1, 0.05, 0.8)
    assert r.n_group_a == r.n_group_b
    assert r.achieved_power >= 0.8
    assert 30 <= r.n_group_a <= 40


def test_proportions_and_anova():
    r = two_proportions(0.3,0.5)
    assert r.n_group_a > 0 and r.achieved_power >= 0.8
    r2 = one_way_anova(0.25,3)
    assert r2.n_group_a >= 2 and r2.achieved_power >= 0.8


def test_ar1_independence_and_correlation():
    assert ar1_effective_frames(1000,0.0) == 1000
    assert math.isclose(ar1_effective_frames(1,.9),1.)
    assert 1 < ar1_effective_frames(1000,.8) < 1000
    assert ar1_effective_frames(1000,.9) < ar1_effective_frames(1000,.5)


def test_deeplabcut_frames_not_independent_animals():
    r = deeplabcut_power(0.5,1,2,1000,3,.8)
    assert r['result'].n_group_a >= 60
    assert r['result'].achieved_power >= .80
    assert r['effective_frames_per_session'] < 1000
    r2 = deeplabcut_power(0.5,1,2,10000,3,.8)
    assert r2['result'].n_group_a >= 60


def test_donor_level_sc():
    r = sc_donor_power(.5,1,2,2000,.1)
    assert r['result'].n_group_a > 0
    assert r['expected_target_cells'] == 200


def test_external_validation_precision():
    o = validation_precision(.8,.9,.3,.05,.95)
    assert wilson_half_width(.8,o['positive_cases'],.95) <= .05
    assert wilson_half_width(.9,o['negative_cases'],.95) <= .05
    assert o['total_recruitment']*.3 >= o['positive_cases']
    assert o['total_recruitment']*.7 >= o['negative_cases']


def test_attrition():
    assert recruitment_n(50,0.1) == 56
    with pytest.raises(ValueError): recruitment_n(20,1)
