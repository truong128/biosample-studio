# BioSample Studio 🧬

An open-source **Streamlit** application for transparent, evidence-informed **sample-size and statistical-power planning** in biological experiments, bulk omics, single-cell/spatial studies, DeepLabCut behavior studies, and independent external validation of a locked binary classifier.

> **Research-planning software, not clinical software.** The validity of every estimate depends on its assumptions. The omics simulator is explicitly an asymptotic Gaussian z-statistic Monte Carlo approximation to gene/feature-level power with Benjamini-Hochberg FDR, **not** a count-level implementation of DESeq2, edgeR, PROPER or RnaSeqSampleSize. This repository does not claim universal or validated sample sizes.

## Modules

| Tab | Endpoint | Independent unit | Engine |
|---|---|---|---|
| Classical | Continuous, paired, binary, one-way ANOVA | Participant / biological replicate / pair | statsmodels noncentral t, proportion z, F |
| Bulk RNA-seq and omics | FDR-controlled feature discovery | Biological replicate | Monte Carlo normal test statistics with NB logFC variance approximation or standardized Gaussian effects + BH |
| Single-cell/spatial | **One** donor-summary endpoint | Donor/specimen | Between- and within-donor Gaussian hierarchical approximation |
| DeepLabCut | **One** behavior summary per animal | Animal or actual independent experimental unit | AR(1)-adjusted frame averaging + animal-level t power |
| External ML validation | Precision of sensitivity and specificity | External patient | Wilson score interval planning |
| References | Method guide | -- | Curated original papers/DOIs |

## Deploy as a free Streamlit Community Cloud website

### Option A: GitHub website (no computer terminal required)

1. Download and unzip the `biosample_studio.zip` package.
2. Log in to https://github.com and create a new repository (e.g. `biosample-studio`).
3. Upload **the contents of** the `biosample_studio` directory into the repository root. `app.py`, `requirements.txt`, `core/`, and `.streamlit/` must appear at the top level. GitHub's website does not always retain hidden `.streamlit` folders in drag-and-drop; check it is present and recreate the folder and its `config.toml` using GitHub's **Add file → Create new file** if necessary.
4. Visit https://share.streamlit.io and sign in with GitHub.
5. Select **Create app → Deploy a public app from GitHub**, then choose your repository, your default branch (typically `main`), and `app.py` as the entrypoint.
6. Click **Deploy**. Streamlit Community Cloud installs `requirements.txt` and produces a shareable HTTPS URL once the build succeeds.

### Option B: Git push

```bash
git init
git add .
git commit -m "Launch BioSample Studio"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/biosample-studio.git
git push -u origin main
```

For existing repositories, omit `git init` and use your established workflow.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11/3.12 recommended. The package uses common Python dependencies and does not require R for the built-in approximation methods.

## Tests

```bash
pip install pytest
python -m pytest -q
python -m compileall -q app.py core tests references.py
```

## Important scientific boundaries

- **Independent units first**: 100,000 RNA-seq single cells from three donors ≠ 100,000 biological replicates; 100,000 pose frames from five mice ≠ 100,000 animals.
- **Omics**: test statistics are independent Gaussian draws with noncentrality chosen from NB delta-method or standardized effect; count-level FDR, gene correlation, outliers, covariates, empirical dispersion-estimation error, and read depth are omitted. Observed FDR is a simulation diagnostic and not assurance of real-data FDR control.
- **Profiling CSV**: A reference profile requires `mean_count,dispersion`, where mean is expected RNA-seq *gene counts* under a comparable library size and dispersion is NB overdispersion. Gene-level pilot expression + dispersion should be obtained from appropriate matched data and tested in the final real statistical pipeline.
- **Proteomics**: logged intensities / standardized continuous endpoints are an approximation. Analyte missingness, per-peptide variance and technical batches warrant MSstats-style pilot modeling.
- **scRNA**: within-donor cell noise is reduced by additional cells, but between-donor heterogeneity remains. Real gene-level differential-state studies should be assessed using scPower or muscat with donor-level pseudobulk or mixed models.
- **DeepLabCut**: the tracking network predicts poses. The study analyzes prespecified animal-level behavioral outcomes. AR1 within-frame correction is only approximate; mixed models or simulations may be essential.
- **ML**: validation-size precision here does **not** provide a development/training sample size. See Riley et al. (2020) and `pmsampsize` for predictor-model development.
- **Data handling**: no application-level account management, custom analytics, external API, or model calls in this app. Streamlit and its hosting provider may use their own cookies and server logs. User-uploaded CSVs are processed on the hosting Streamlit server. Avoid uploading patient-identifiable files to publicly hosted instances. No persistent database is configured.

## References and methods

See [docs/METHODS.md](docs/METHODS.md) and the **Methods & References** page of the deployed app for original papers, formulas, and DOI links. The sources are a curated selection of methodology, **not a systematic review of every paper published in the field**.

## License

MIT. Attribution to cited scientific sources is independent of the software license.
