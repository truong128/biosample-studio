"""BioSample Studio: Streamlit sample-size planning for life science studies.

Run locally: streamlit run app.py
This is design-stage guidance; statistical consultation recommended for confirmatory work.
"""
import json
from dataclasses import asdict
from datetime import datetime, timezone
from math import ceil

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.omics import OmicsConfig, profile_from_csv, recommend_n, run_omics_curve
from core.power import (
    deeplabcut_power, one_way_anova, paired_t, power_curve_t,
    recruitment_n, sc_donor_power, two_proportions, two_sample_t,
    validation_precision,
)
from references import REFERENCES, get_refs

APP_NAME = 'BioSample Studio'
VERSION = '1.0.0'

st.set_page_config(page_title=f'{APP_NAME} | Sample Size & Power', page_icon='🧬',
                   layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container {padding-top: 1.8rem; max-width: 1300px;}
h1,h2,h3 {letter-spacing: -.025em;}
h1 {font-size: 2.3rem !important;}
.hero {background: linear-gradient(112deg, #12334f 0%,#1c526b 58%,#0e7371 100%);
       padding: 2.0rem 2.1rem; border-radius: 18px; color: white; margin-bottom: 1rem;}
.hero h1 {color:white !important; margin:0 0 .25rem !important;}
.hero p {color:#e2f4fa; font-size:1.06rem; margin:0;}
.smallnote {background:#f4f7fa; border-left:4px solid #137a8c; padding:.8rem 1rem;
            border-radius:4px; color:#27394a; margin-top:.4rem; margin-bottom:.7rem;}
div[data-testid="stMetric"] {background:#f1f7fa; padding:12px 16px; border:1px solid #ddebf0; border-radius:12px;}
</style>''', unsafe_allow_html=True)


def refs_show(*keys):
    st.markdown('#### Cơ sở khoa học / References')
    for ref in get_refs(*keys):
        st.markdown(f"- [{ref['citation']}]({ref['url']})")


def result_report(module, params, result, notes, citations):
    return {'software': APP_NAME, 'version': VERSION,
            'generated_utc': datetime.now(timezone.utc).isoformat(),
            'module': module, 'assumptions': params, 'results': result,
            'limitations': notes, 'references': [ref['url'] for ref in get_refs(*citations)]}


def deliver_report(report, label='study'):
    document = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    st.download_button('⬇️ Tải báo cáo JSON (giả định, kết quả, nguồn)', document,
                       file_name=f'biosample_{label}_report.json', mime='application/json',
                       use_container_width=True)
    lines = [f'# BioSample Studio | {report["module"]}', '',
             'Generated (UTC): '+report['generated_utc'], '',
             '## Inputs / Assumptions', '']
    lines += [f'- **{k}**: {v}' for k,v in report['assumptions'].items()]
    lines += ['', '## Results','']
    lines += [f'- **{k}**: {v}' for k,v in report['results'].items()]
    lines += ['', '## Limitations','', report['limitations'], '', '## References','']
    lines += [f'- {url}' for url in report['references']]
    st.download_button('⬇️ Tải Methods (Markdown)', '\n'.join(lines),
                       file_name=f'biosample_{label}_methods.md', mime='text/markdown',
                       use_container_width=True)


def power_plot(ns, vals, target, x_label='Biological samples per group'):
    df = pd.DataFrame({'n': ns, 'power': vals})
    fig = px.line(df, x='n', y='power', markers=True, labels={'n':x_label,'power':'Statistical power'})
    fig.add_hline(y=target, line_dash='dash', annotation_text=f'Target {target:.0%}')
    fig.update_yaxes(range=[0,1.025], tickformat='.0%')
    fig.update_layout(height=375, margin=dict(t=25,r=15,l=10,b=20))
    st.plotly_chart(fig, use_container_width=True)


def show_sample_metrics(n_a, n_b, achieved, attrition, unit='mẫu sinh học'):
    c1,c2,c3,c4=st.columns(4)
    c1.metric(f'{unit} · nhóm A', f'{n_a:,}')
    c2.metric(f'{unit} · nhóm B', f'{n_b:,}')
    c3.metric('Power ước tính', f'{achieved:.1%}')
    c4.metric('Tổng cần tuyển sau hao hụt',
              f'{recruitment_n(n_a, attrition) + recruitment_n(n_b, attrition):,}')


st.sidebar.markdown('### 🧬 BioSample Studio')
st.sidebar.caption('Evidence-informed sample size planning | v'+VERSION)
PAGES = [
    '🏠 Bắt đầu / Study planner',
    '🧪 Sinh học cơ bản / Classical',
    '🧬 Bulk RNA-seq & Omics FDR',
    '🔬 Single-cell / Spatial',
    '🐁 DeepLabCut / Behavior',
    '🎯 External ML validation',
    '📚 Methods & References',
]
page = st.sidebar.radio('Chọn bộ tính', PAGES, label_visibility='collapsed')
st.sidebar.divider()
st.sidebar.info('⚠️ Kết quả phụ thuộc giả định. Không thay thế đánh giá của chuyên gia biostatistics, hoặc tính power trên pilot data phù hợp.')
st.sidebar.caption('No user account, database, tracking, or remote API in this source. Uploaded CSV data are processed by the Streamlit server hosting the app.')

st.markdown('''<div class="hero"><h1>BioSample Studio</h1>
<p>Scientific sample size and power planning for biomedical experiments, omics, single-cell and markerless behavior tracking.</p></div>''', unsafe_allow_html=True)

if page == PAGES[0]:
    st.markdown('## Chọn thiết kế trước khi tính sample size')
    st.markdown('Phần mềm phân biệt **biological replicate** với phép đo lặp lại và hỗ trợ báo cáo các giả định cần thiết khi viết protocol, manuscript hoặc grant.')
    col1,col2,col3 = st.columns(3)
    with col1:
        st.markdown('### 🧪 Sinh học cơ bản')
        st.write('Hai nhóm độc lập, paired trước-sau, hai tỷ lệ, ANOVA một yếu tố. Power theo mô hình t, z hoặc F.')
        st.markdown('### 🧬 RNA-seq / Omics')
        st.write('Ước lượng số mẫu độc lập mỗi nhóm để đạt power khi kiểm soát BH-FDR trên hàng nghìn đặc trưng.')
    with col2:
        st.markdown('### 🔬 Single-cell / Spatial')
        st.write('Số donor, số tế bào/donor và biến thiên giữa donor; không dùng số tế bào làm n độc lập.')
        st.markdown('### 🐁 DeepLabCut')
        st.write('Số động vật, session, frame, tự tương quan thời gian và repeated measurements.')
    with col3:
        st.markdown('### 🎯 Machine learning')
        st.write('Độ chính xác của sensitivity/specificity khi đánh giá trên một external test cohort; không tính cỡ mẫu huấn luyện mô hình.')
        st.markdown('### 📚 Reference library')
        st.write('Nguồn primary-method đã lựa chọn, DOI và minh bạch về phạm vi áp dụng.')
    st.divider()
    st.subheader('5 nguyên tắc trước khi quyết định số mẫu')
    st.markdown('''1. **Primary endpoint:** Chọn chính xác phép đo/hypothesis dùng làm quyết định chính.
2. **Experimental unit:** Đơn vị độc lập được randomize hoặc lấy mẫu. Frame, cell, lát cắt và technical replicate thường không độc lập.
3. **Minimum relevant effect:** Khác biệt nhỏ nhất có giá trị sinh học hoặc lâm sàng.
4. **Variance:** Ước lượng từ nghiên cứu phù hợp, pilot đủ dữ liệu, hoặc phân tích độ nhạy nhiều giả định.
5. **Power, alpha, FDR:** Đặt trước phân tích, có kế hoạch QC, batch, missingness và recruitment inflation.''')
    st.warning('Không có một cỡ mẫu cố định áp dụng được cho mọi RNA-seq, every proteomics assay, mỗi mô hình ML hay DeepLabCut. Các module dùng mức xấp xỉ khác nhau, hãy đọc giới hạn riêng.')
    refs_show('ARRIVE_EE','PROPER','SCPOWER','PSEUDO')

elif page == PAGES[1]:
    st.header('Classical biological experiment calculator')
    kind = st.selectbox('Thiết kế', ['Hai nhóm độc lập (continuous)', 'Paired / repeated before-after',
                                     'Hai tỷ lệ (binary endpoint)', 'One-way ANOVA (balanced groups)'])
    with st.form('classical_form'):
        c1,c2,c3=st.columns(3)
        with c1: alpha=st.number_input('Two-sided alpha', .001,.20,.05,.005, format='%.3f')
        with c2: target=st.slider('Target power', .50,.99,.80,.01)
        with c3: attrition=st.slider('Expected sample exclusion / attrition',0.,.50,.10,.01)
        if kind == 'Hai nhóm độc lập (continuous)':
            c1,c2,c3=st.columns(3)
            with c1: delta=st.number_input('Absolute difference of group means (unit)', min_value=.001,value=.5,step=.1)
            with c2: sd=st.number_input('Within-group standard deviation (same unit)',min_value=.001,value=1.,step=.1)
            with c3: ratio=st.number_input('Group B / Group A ratio',min_value=.2,max_value=5.,value=1.,step=.1)
        elif kind == 'Paired / repeated before-after':
            c1,c2=st.columns(2)
            with c1: delta=st.number_input('Mean paired difference',min_value=.001,value=.5,step=.1)
            with c2: sd_diff=st.number_input('SD of the *within-subject difference*',min_value=.001,value=1.,step=.1)
        elif kind == 'Hai tỷ lệ (binary endpoint)':
            c1,c2,c3=st.columns(3)
            with c1: pa=st.number_input('Group A event proportion',.001,.999,.30,.01)
            with c2: pb=st.number_input('Group B event proportion',.001,.999,.50,.01)
            with c3: ratio=st.number_input('Group B / Group A ratio',.2,5.,1.,.1)
        else:
            c1,c2=st.columns(2)
            with c1: f=st.number_input('Cohen f (standardized between-group effect)',.01,3.,.25,.05)
            with c2: groups=st.number_input('Number of groups',min_value=3,max_value=20,value=3,step=1)
        submitted=st.form_submit_button('Tính sample size & power',use_container_width=True,type='primary')
    if submitted:
        try:
            if kind.startswith('Hai nhóm'):
                r=two_sample_t(delta,sd,alpha,target,ratio)
            elif kind.startswith('Paired'):
                r=paired_t(delta,sd_diff,alpha,target)
            elif kind.startswith('Hai tỷ'):
                r=two_proportions(pa,pb,alpha,target,ratio)
            else:
                r=one_way_anova(f,groups,alpha,target)
            st.session_state['classical_result']=(kind,r,attrition,alpha,target,{})
        except (ValueError,OverflowError,RuntimeError) as e:
            st.error(f'Input hoặc giải nghiệm không phù hợp: {e}')
    if 'classical_result' in st.session_state:
        saved_kind,r,loss,a,p,_=st.session_state['classical_result']
        if saved_kind != kind:
            st.info('Thiết kế vừa thay đổi; nhấn Tính để tạo kết quả cho thiết kế mới.')
        else:
            if kind.startswith('One-way'):
                st.metric('Mỗi nhóm',r.n_group_a)
                st.metric('Tổng mẫu có thể phân tích',r.n_group_b)
                st.metric('Tổng mẫu tuyển (ước tính)',recruitment_n(r.n_group_a,loss)*int(groups))
            elif kind.startswith('Paired'):
                st.metric('Số người/động vật có cả hai thời điểm',r.n_group_a)
                st.metric('Cần tuyển (đã xét hao hụt)',recruitment_n(r.n_group_a,loss))
            else:
                show_sample_metrics(r.n_group_a,r.n_group_b,r.achieved_power,loss)
            st.caption(f'Method: {r.method}. Standardized effect size = {r.effect_size:.3f}.')
            if not kind.startswith('One-way') and not kind.startswith('Hai tỷ'):
                ns=np.unique(np.maximum(3, np.round(np.linspace(max(3,r.n_group_a//4),
                                              max(10,r.n_group_a*2),30)).astype(int)))
                values=power_curve_t(r.effect_size,a,ns,paired=kind.startswith('Paired'),
                                     ratio=1 if kind.startswith('Paired') else r.n_group_b/r.n_group_a)
                power_plot(ns,values,p,'Pairs' if kind.startswith('Paired') else 'Subjects per group A')
            notes='Requires independent sampling units and its stated parametric test assumptions. Welch unequal-variance, covariate-adjusted, cluster and longitudinal models require additional design-specific analysis.'
            st.info(notes)
            params={'kind':kind,'alpha':a,'target_power':p,'attrition':loss,'standardized_effect_size':r.effect_size}
            deliver_report(result_report('Classical',params,asdict(r),notes,['ARRIVE_EE']), 'classical')
    refs_show('ARRIVE_EE')

elif page == PAGES[2]:
    st.header('Bulk RNA-seq and high-dimensional omics')
    st.markdown('**FDR-aware Monte Carlo power analysis** using asymptotic normal test statistics. This is a fast **planning approximation**, not an implementation of DESeq2, edgeR, RnaSeqSampleSize or PROPER.')
    assay_label=st.radio('Modality / statistical endpoint',
          ['RNA-seq / ATAC-seq gene-or-peak counts (NB approximation)',
           'Proteomics / metabolomics / log-intensity features (standardized effect)'],
          horizontal=True)
    assay='bulk_rnaseq' if assay_label.startswith('RNA') else 'standardized'
    upload=None
    if assay=='bulk_rnaseq':
        with st.expander('Optional: upload feature-wise reference profile (CSV)', expanded=False):
            st.caption('CSV columns: `mean_count,dispersion` (one row per feature; 100-25,000 rows). Derived from independent comparable public/pilot count data, not TPM/FPKM. This tool does not estimate dispersion directly from a count matrix.')
            st.code('mean_count,dispersion\n10,0.3\n50,0.2\n120,0.14\n...', language='text')
            upload=st.file_uploader('Select profile CSV',type=['csv'],key='pilot_uploader')
    with st.form('omics_form'):
        cols=st.columns(4)
        with cols[0]: fdr=st.selectbox('Benjamini-Hochberg FDR', [.01,.05,.10],index=1)
        with cols[1]: target=st.slider('Target discovery power',.50,.99,.80,.01)
        with cols[2]: pi1=st.slider('Assumed true DE fraction',.005,.5,.05,.005,format='%.3f')
        with cols[3]: attrition=st.slider('Expected QC failure',0.,.50,.10,.01)
        if assay=='bulk_rnaseq':
            cols=st.columns(4)
            with cols[0]: fc=st.number_input('Minimum target fold change',1.05,10.,1.5,.05)
            with cols[1]: median=st.number_input('Median expected count',1.,100000.,50.,5.)
            with cols[2]: disp=st.number_input('Median NB dispersion φ',.001,5.,.20,.02)
            with cols[3]: lcs=st.number_input('Log-SD across means',0.,3.,.6,.1)
            lds=st.slider('Log-SD of gene dispersions',0.,2.,.35,.05)
            deff=.65
        else:
            deff=st.number_input('Target standardized effect (Cohen d)',.05,5.,.65,.05)
            fc=1.5;median=50.;disp=.2;lcs=.6;lds=.35
        cols=st.columns(4)
        with cols[0]: features=st.number_input('Number of tested features',100,25000,4000,100)
        with cols[1]: repetitions=st.number_input('Monte Carlo repetitions',10,2000,120,10)
        with cols[2]: seed=st.number_input('Random seed',0,2147483647,20261027,1)
        with cols[3]: n_grid=st.text_input('n per group (comma separated)', '5,8,10,12,15,20,25,30,40,50,60,80,100')
        submit=st.form_submit_button('▶ Run BH-FDR power simulation',type='primary',use_container_width=True)
    if submit:
        try:
            ns=[int(x.strip()) for x in n_grid.split(',') if x.strip()]
            config=OmicsConfig(assay=assay,features=int(features),pi1=pi1,fold_change=fc,
                     median_count=median,dispersion=disp,log_count_sd=lcs,
                     log_dispersion_sd=lds,standardized_effect=deff,fdr=fdr,
                     target_power=target,repetitions=int(repetitions),seed=int(seed))
            pilot=profile_from_csv(upload.getvalue(),assay) if upload else None
            with st.spinner('Simulating high-dimensional power and BH FDR...'):
                curve=run_omics_curve(config,ns,pilot)
            st.session_state['omics_result']=(config,curve,pilot is not None,float(attrition),len(pilot) if pilot is not None else config.features)
        except (ValueError,OverflowError) as ex:
            st.error(f'Unable to run simulation: {ex}')
    if 'omics_result' in st.session_state:
        c,curve,pilot_flag,loss,features_used=st.session_state['omics_result']
        if c.assay!=assay:
            st.info('Các kết quả hiển thị đang thuộc modality khác. Chạy lại simulation để cập nhật.')
        else:
            n_reco=recommend_n(curve)
            c1,c2,c3=st.columns(3)
            c1.metric('Suggested n/group (tested grid)',str(n_reco) if n_reco else 'Not reached')
            c2.metric('Genome-wide BH threshold',f'{c.fdr:.0%}')
            c3.metric('Assumed true DE fraction',f'{c.pi1:.1%}')
            st.caption(f'Actual features simulated: {features_used:,}; uploaded reference profile: {pilot_flag}.')
            if n_reco:
                st.success(f'Trong các n đã thử, {n_reco} biological samples/group là lựa chọn đầu tiên đạt target power; tuyển ≥ {recruitment_n(n_reco,loss)} mẫu/group với QC loss {loss:.0%}. Đây không phải giá trị minimum chính xác.')
            else:
                st.warning('Chưa đạt target power trong dãy n đã thử. Mở rộng grid và/hoặc xem lại effect size và variance assumptions.')
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=curve['n_per_group'],y=curve['expected_power'],
                mode='lines+markers',name='Expected targeted power'))
            fig.add_trace(go.Scatter(x=list(curve['n_per_group'])+list(curve['n_per_group'][::-1]),
                y=list(curve['power_CI_high_MC'])+list(curve['power_CI_low_MC'][::-1]),
                fill='toself',fillcolor='rgba(13,135,157,.13)',line=dict(color='rgba(0,0,0,0)'),
                name='Monte Carlo 95% CI'))
            fig.add_hline(y=c.target_power,line_dash='dash',annotation_text='Target power')
            fig.update_layout(title='BH-FDR discovery power by biological n/group',
                       xaxis_title='n independent biological samples/group',yaxis_title='Power',
                       yaxis_range=[0,1.025],height=405)
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(curve.style.format({'expected_power':'{:.1%}',
             'power_CI_low_MC':'{:.1%}','power_CI_high_MC':'{:.1%}',
             'mean_false_discovery_proportion':'{:.2%}', 'mean_discoveries':'{:.1f}',
             'mean_false_positive_calls':'{:.1f}'}),hide_index=True,use_container_width=True)
            st.download_button('⬇️ Tải kết quả toàn bộ grid (.csv)',curve.to_csv(index=False),
                              file_name='biosample_omics_power_curve.csv',mime='text/csv')
            notes=('Z-statistic Monte Carlo under assumed independent features and correctly calibrated null p-values; '
                   'bulk RNA-seq variance is based on the NB delta-method. Does NOT generate RNA counts or fit DESeq2, '
                   'model real sample-level batch effects, gene correlation, estimated dispersion, fold-change-threshold '
                   'hypothesis testing, cell composition, or sequencing read depth. MC interval reflects simulation error '
                   'only, not uncertainty in biological assumptions. Pilot reference profile is not a substitute for '
                   'pipeline-matched counts-based simulation. Confirm final n using PROPER or RnaSeqSampleSize with '
                   'appropriate pilot data and the intended differential-expression pipeline.')
            st.warning(notes)
            report=result_report('Omics FDR planning',asdict(c),
                    {'tested_grid':curve.to_dict(orient='records'),'first_n_in_grid_reaching_target':n_reco,
                     'profile_uploaded':pilot_flag,'features_actually_simulated':features_used,'recruitment_per_group_if_reached':recruitment_n(n_reco,loss) if n_reco else None},
                    notes,['BH','DESEQ2','PROPER','RNA_SS','MSSTATS'])
            deliver_report(report,'omics')
    refs_show('BH','DESEQ2','PROPER','RNA_N','RNA_SS','MSSTATS')

elif page == PAGES[3]:
    st.header('Single-cell and spatial: biological donors vs cells / spots')
    st.info('Chỉ tính cho **một endpoint** là trung bình ở cấp donor / biological specimen. Đối với DE hàng nghìn gene, cần scPower, muscat hoặc simulation theo cell type / pseudobulk; kết quả ở đây KHÔNG có điều chỉnh FDR.')
    with st.form('sc_form'):
        c1,c2,c3=st.columns(3)
        with c1: delta=st.number_input('Target difference of donor summaries',.001,10.,.5,.1)
        with c2: donor_sd=st.number_input('Between-donor SD',.001,20.,1.,.1)
        with c3: cell_sd=st.number_input('Within-donor (cell/spot) SD',.001,20.,2.,.1)
        c1,c2,c3=st.columns(3)
        with c1: cells=st.number_input('Total cells/spots sampled per donor',100,500000,2000,100)
        with c2: fraction=st.number_input('Fraction of target cell type / region',.0001,1.,.10,.01,format='%.4f')
        with c3: attrition=st.slider('Donor QC failure / dropout',0.,.5,.1,.01)
        c1,c2=st.columns(2)
        with c1: alpha=st.number_input('Nominal alpha (single endpoint)',.001,.2,.05,.005)
        with c2: target=st.slider('Target power',.5,.99,.8,.01)
        submit=st.form_submit_button('Calculate independent donors',use_container_width=True,type='primary')
    if submit:
        try:
            out=sc_donor_power(delta,donor_sd,cell_sd,cells,fraction,alpha,target)
            st.session_state['sc_result']=(out,attrition,dict(delta=delta,between_donor_SD=donor_sd,
             within_donor_SD=cell_sd,total_cells_per_donor=cells,target_fraction=fraction,
             alpha=alpha,power=target,attrition=attrition))
        except (ValueError,OverflowError) as e:
            st.error(str(e))
    if 'sc_result' in st.session_state:
        o,loss,inp=st.session_state['sc_result']
        r=o['result']
        show_sample_metrics(r.n_group_a,r.n_group_b,r.achieved_power,loss,'Donors')
        c1,c2=st.columns(2)
        c1.metric('Expected usable target cells/donor',f'{o["expected_target_cells"]:,.1f}')
        c2.metric('SD of donor-level summary',f'{o["sd_donor_summary"]:.3f}')
        if o['expected_target_cells']<10:
            st.warning('Rất ít cells/region thuộc target. Distribution of donor-level means and missing cell types will require explicit simulation.')
        st.warning('Cấp donor là đơn vị độc lập. Hierarchical Gaussian variance approximation only; no gene-level FDR, no dropout/zero inflation, no reads-per-cell, no cell-type detection model, no batch or library depth. Spatial spots may be spatially autocorrelated.')
        deliver_report(result_report('Single-cell donor endpoint',inp,
                        {'n_donors_per_group':r.n_group_a,'power':r.achieved_power,
                         'expected_target_cells':o['expected_target_cells']},
                        'Pilot-based pseudobulk/muscat/scPower analysis needed for DE and multiple testing.',
                        ['SCPOWER','MUSCAT']),'singlecell')
    refs_show('SCPOWER','MUSCAT','ARRIVE_EE')

elif page == PAGES[4]:
    st.header('DeepLabCut: animal-level power for behavioral outcomes')
    st.markdown('DeepLabCut trích xuất vị trí / pose. **Behavioral endpoint** cần được định nghĩa riêng (ví dụ speed, freezing %, bout duration). Đơn vị n thường là động vật được phân nhóm độc lập, **không phải video frames**.')
    with st.form('dlc_form'):
        c1,c2,c3=st.columns(3)
        with c1: delta=st.number_input('Expected difference in animal-level endpoint',.001,1000.,.5,.1)
        with c2: sd_animal=st.number_input('Between-animal SD',.001,1000.,1.,.1)
        with c3: sd_frame=st.number_input('Within-animal frame-level SD',.001,1000.,2.,.1)
        c1,c2,c3=st.columns(3)
        with c1: frames=st.number_input('Usable frames per session',1,200000,3000,100)
        with c2: sessions=st.number_input('Independent sessions per animal',1,100,3,1)
        with c3: rho=st.slider('Adjacent frame correlation (AR1 ρ)',0.,.99,.8,.01)
        c1,c2,c3,c4=st.columns(4)
        with c1: alpha=st.number_input('Familywise alpha',.001,.2,.05,.005)
        with c2: endpoints=st.number_input('Confirmatory behavioral endpoints',1,100,1,1)
        with c3: target=st.slider('Power target',.5,.99,.8,.01)
        with c4: loss=st.slider('Animal dropout/QC',0.,.5,.1,.01)
        submit=st.form_submit_button('Calculate independent animals',use_container_width=True,type='primary')
    if submit:
        try:
            out=deeplabcut_power(delta,sd_animal,sd_frame,frames,sessions,rho,alpha,target,int(endpoints))
            inputs=dict(delta=delta,animal_SD=sd_animal,frame_SD=sd_frame,frames_per_session=int(frames),
                   sessions=int(sessions),rho=rho,alpha=alpha,endpoints=int(endpoints),power=target,loss=loss)
            st.session_state['dlc_result']=(out,inputs)
        except (ValueError,OverflowError) as e:
            st.error(str(e))
    if 'dlc_result' in st.session_state:
        o,inputs=st.session_state['dlc_result']
        r=o['result']
        show_sample_metrics(r.n_group_a,r.n_group_b,r.achieved_power,inputs['loss'],'Animals')
        c1,c2,c3=st.columns(3)
        c1.metric('Frames recorded per animal',f'{o["total_frame_count_per_animal"]:,}')
        c2.metric('Effective frames/session (AR1)',f'{o["effective_frames_per_session"]:,.1f}')
        c3.metric('Bonferroni alpha/endpoint',f'{o["adjusted_alpha"]:.4g}')
        st.warning('Approximation: independent animals and sessions; AR(1) frame autocorrelation; between-animal SD does not vanish with more frames. Assumes behavioral endpoint can be summarized as an approximately normal animal mean. Does NOT account for cage/litter nesting, non-stationary movement, unequal session effects, bounded/count outcomes, DLC tracking errors, or mixed-effects time-by-treatment interactions.')
        deliver_report(result_report('DeepLabCut behavior',inputs,
               {'animals_per_group':r.n_group_a,'power':r.achieved_power,
                'effective_frames_per_session':o['effective_frames_per_session'],
                'total_recruitment':2*recruitment_n(r.n_group_a,inputs['loss'])},
                'One continuous animal-level summary. For repeated-measures mixed models or nested cages/litters simulate intended model.',
                ['DLC','PSEUDO','ARRIVE','ARRIVE_EE']), 'deeplabcut')
    refs_show('DLC','PSEUDO','ARRIVE','ARRIVE_EE')

elif page == PAGES[5]:
    st.header('External classifier validation: precision of sensitivity & specificity')
    st.info('Đây là **sample size cho đánh giá mô hình đã khóa trên tập external test**, không phải sample size cho training ML/AI model. Không ước lượng power để so sánh AUC hay validation across sites.')
    with st.form('validation_form'):
        c1,c2,c3=st.columns(3)
        with c1: sens=st.number_input('Expected sensitivity',.01,.99,.85,.01)
        with c2: spec=st.number_input('Expected specificity',.01,.99,.85,.01)
        with c3: prev=st.number_input('Disease prevalence in external cohort',.001,.999,.25,.01)
        c1,c2=st.columns(2)
        with c1: margin=st.number_input('Maximum Wilson CI half-width',.005,.2,.05,.005)
        with c2: confidence=st.selectbox('Confidence level',[.90,.95,.99],index=1)
        submit=st.form_submit_button('Calculate external validation sample size',use_container_width=True,type='primary')
    if submit:
        try:
            out=validation_precision(sens,spec,prev,margin,confidence)
            st.session_state['validation_result']=(out,dict(sensitivity=sens,specificity=spec,
                                       prevalence=prev,margin=margin,confidence=confidence))
        except (ValueError,OverflowError) as e:
            st.error(str(e))
    if 'validation_result' in st.session_state:
        o,params=st.session_state['validation_result']
        c1,c2,c3=st.columns(3)
        c1.metric('Min positive cases',o['positive_cases'])
        c2.metric('Min negative cases',o['negative_cases'])
        c3.metric('Total external cases (expected)',o['total_recruitment'])
        st.warning('Wilson score interval half-width at an assumed true proportion; expected prevalence determines N. Does not guarantee the exact realized number of positives or negatives. Use stratified enrollment or prevalence uncertainty sensitivity analysis. Development sample-size requirements differ: see Riley et al.')
        deliver_report(result_report('External ML model validation',params,o,
          'External test cohort precision only; not adequate for determining training sample size.',
          ['RILEY']),'validation')
    refs_show('RILEY')

else:
    st.header('Methods, validity tiers and literature')
    st.markdown('''**Mức độ của các module**

| Module | Statistical implementation | Validity / what is excluded |
|---|---|---|
| Classical | Noncentral t, normal two-proportion, ANOVA F power | Assumes proper sampling units and its stated test |
| RNA-seq | Asymptotic NB logFC variance → Gaussian z → simulated BH | **Exploratory only**; not fitted DESeq2 / edgeR; no count-level empirical calibration |
| Proteomics/metabolomics | Standardized Gaussian z → simulated BH | Exploratory, no missingness / peptide or protein-specific effects |
| scRNA / Spatial | Donor-summary normal variance, between + within donor | One endpoint, no pseudobulk count-FDR or spatial correlation |
| DeepLabCut | Animal-summary t test with AR1 effective frame count | Not a hierarchical mixed model; no cage/litter, nonlinear behavior, tracking error |
| ML external validation | Wilson CI precision of sensitivity and specificity | Does not size development/training cohorts |

**Không nên hiểu** các tham khảo là phần mềm này đã thực thi đầy đủ thuật toán của từng bài báo. Những tài liệu này xác định phương pháp khoa học, giả định, hoặc các phương pháp chuyên biệt mà người nghiên cứu nên dùng để confirm.
''')
    st.markdown('### Tham khảo có liên kết DOI / tài liệu gốc')
    st.dataframe(pd.DataFrame([{'Topic':r['topic'],'Paper / method':r['citation'],
                               'DOI':r['doi'], 'URL':r['url']} for r in REFERENCES]),
                 hide_index=True,use_container_width=True)
    st.download_button('⬇️ Download bibliography (.csv)',pd.DataFrame(REFERENCES).to_csv(index=False),
                       file_name='biosample_references.csv',mime='text/csv')
    st.subheader('Pre-registration & reporting checklist')
    for item in [
        'Chọn primary endpoint và minimum biologically meaningful effect trước phân tích',
        'Định nghĩa biological unit và unit of randomization',
        'Lấy variance assumptions từ nghiên cứu gần nhất hoặc multiple plausible scenarios',
        'Prespecify alpha, target power, FDR hoặc multiplicity correction và group allocation',
        'Xác định batch, sex, age, tissue composition, cage/litter hoặc donor clustering',
        'Lập kế hoạch exclusions, missing assays, low-quality tracking và hao hụt mẫu',
        'Viết rõ software version, seed, simulation grid, repetitions, method limitations',
        'Với confirmatory omics: calibrate against pilot/public data and intended analysis pipeline',
    ]:
        st.checkbox(item,value=False)
    st.caption('Research software / exploratory research planning. It is not a clinical decision support device.')

st.divider()
st.caption('BioSample Studio v'+VERSION+' | Open-source Streamlit application | Conditional estimates; not guaranteed sample sizes | No claims of universal optimal n.')
