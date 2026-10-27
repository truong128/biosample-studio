import io
import numpy as np
import pandas as pd
import pytest
from core.omics import OmicsConfig, _bh_rejections, profile_from_csv, run_omics_curve, recommend_n


def test_bh_mask_honors_stepup_rule():
    p=np.array([.001,.03,.02,.9])
    mask=_bh_rejections(p,.05)
    assert mask.tolist()==[True,True,True,False]


def test_omics_curve_is_deterministic():
    conf=OmicsConfig(features=300,pi1=.1,fold_change=2,median_count=50,
                     dispersion=.2,repetitions=30,seed=42)
    df=run_omics_curve(conf,[8,12,40])
    df2=run_omics_curve(conf,[8,12,40])
    pd.testing.assert_frame_equal(df,df2)
    assert 0 <= df['mean_false_discovery_proportion'].max() <=1
    assert df.loc[df.n_per_group.eq(40),'expected_power'].iloc[0] > df.loc[df.n_per_group.eq(8),'expected_power'].iloc[0]


def test_omics_zero_target_not_returned_if_all_below():
    conf=OmicsConfig(features=200,pi1=.05,fold_change=1.05,dispersion=1,
                     repetitions=20,seed=10)
    df=run_omics_curve(conf,[5,10])
    assert recommend_n(df) is None


def test_standardized_omics():
    conf=OmicsConfig(assay='standardized', features=200,pi1=.1,
                     standardized_effect=.75,repetitions=30,seed=4)
    curve=run_omics_curve(conf,[10,30,70])
    assert curve.expected_power.iloc[-1] > curve.expected_power.iloc[0]


def test_csv_validation():
    csv=('mean_count,dispersion\n'+''.join('50,0.2\n' for _ in range(100))).encode()
    p=profile_from_csv(csv,'bulk_rnaseq')
    assert len(p)==100
    with pytest.raises(ValueError): profile_from_csv(b'bad,dispersion\n1,2\n','bulk_rnaseq')
