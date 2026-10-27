# Methods and validation notes

## 1. Scientific hierarchy of experimental units

Independent units are set by biological sampling/randomization, not by number of observations. Key guidance: ARRIVE 2.0; Lazic (2010) on pseudoreplication. Count the treatment-assignment/randomization unit (often participant, animal, litter, or cage). Technical replicates, frames, regions and cells within one unit are correlated.

## 2. Conventional test sample sizes

Two-group standardized effect d = |muB - muA|/sigma and noncentral t power for independent equal-variance groups; paired d_z = |mean paired difference|/SD(paired differences). Group ratio is nB/nA. One-way ANOVA uses Cohen f and number of groups; total N is rounded to a balanced n per group. Two binary proportions use Cohen's h and normal approximation. Test assumptions matter, especially for low prevalences, heteroscedasticity, clusters, repeated measures, or rare outcomes. `statsmodels.stats.power` implements calculations.

## 3. Bulk RNA-seq and omics

RNA-seq count NB mean/dispersion variance `Var(Y)=mu+phi*mu^2`. With equal sample sizes n per arm, delta-method variance of log(meanB/meanA) approximately `(1/muA+phi)/n + (1/muB+phi)/n`. For each simulated feature, asymptotic z statistic has null `N(0,1)`, and under alternative `N(log(FC)/SE,1)`. The user specifies true fraction of non-null features (pi1), count/dispersion distributions, minimum target fold change, features tested, and power/FDR thresholds. Each Monte Carlo replicate draws independent z statistics (not RNA-seq reads), computes two-sided p values, and applies Benjamini-Hochberg. Detection power is TP / truly alternative features; mean false discovery proportion is mean `FP/max(number_called,1)` across studies. Non-null features are randomly selected and are modeled as sharing the selected effect size. Because independent normals are simulated, BH's idealized theoretical assumptions hold, but this does **not** model DESeq2/edgeR estimation or gene dependence. The MC confidence interval quantifies simulation error given fixed model inputs, not uncertainty in disease/tissue variability.

For a validated study-specific omics plan, obtain comparable expression/dispersion profiles, fit the intended count pipeline (DESeq2, edgeR, PROPER, RnaSeqSampleSize), incorporate batch covariates, simulate counts at each n, and empirically estimate actual FDR and power. For methods testing `|log2FC| > threshold`, simulate the actual thresholded hypothesis test rather than ordinary H0: logFC=0 used here.

For continuous high-dimensional intensities, assume standardized effect d, asymptotic z alternative mean d*sqrt(n/2). This is a Gaussian approximation to two independent equal-variance group means and models BH correction, **not** protein peptide level data or missingness. MSstats provides variance-calibrated proteomics sample-size planning.

## 4. scRNA-seq / spatial (one prechosen continuous donor endpoint only)

A donor-level summary has assumed SD `sqrt(sigma_between^2 + sigma_within^2 / (total_cells_per_donor * target_fraction))`. Power then uses an unpaired t test across biological donors. This ignores correlations among cells within donor beyond a single residual variance term, cell-type absences, sequencing depth, compositional effects, feature-level FDR and spatial autocorrelation. The relevant formal frameworks are scPower and muscat.

## 5. DeepLabCut (one prechosen continuous behavioral endpoint only)

For stationary adjacent AR(1) frame correlation rho, the finite-sample variance of a frame mean is proportional to `[m + 2 sum_{k=1}^{m-1}(m-k)rho^k]/m^2`; its effective frames `m_eff = m^2/[m + 2 sum (m-k)rho^k]`. If sessions are independent within animal conditional on its baseline, assumed animal-summary SD is `sqrt(sigma_between_animal^2 + sigma_frame^2/(sessions*m_eff))`. Independent animals determine t-test sample size. Multiple confirmatory outcomes use a conservative Bonferroni adjustment. Frames are not independent animals. For longitudinal behavior (trajectories), batches, repeated sessions with shared session effects, cage/litter clustering or treatment-by-time interaction, plan using a hierarchical mixed-effects simulation using pilot tracking output.

## 6. ML external validation

For expected sensitivity/specificity and a target two-sided Wilson score interval *half-width*, binary search gives the minimum **per-class number** meeting the target under expected performance. Convert cases/controls into expected total sample size based on prevalence. This is not a sample size method for ML training or AUC comparative tests; clinical prediction-model development is governed by separate methods, such as Riley et al. (BMJ 2020).

## 7. Reporting recommendation

Always include primary endpoint, experimental unit, effect assumptions, variance provenance, n/group, statistical model, multiplicity method, alpha/FDR, target power, losses/QC, sensitivity analysis, random seed, software version, and limitations. Place pilot/calibration source and tissue context in the final protocol.

## Primary references

1. Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
2. Love MI, Huber W, Anders S (2014). DESeq2. https://doi.org/10.1186/s13059-014-0550-8
3. Wu H, Wang C, Wu Z (2015). PROPER. https://doi.org/10.1093/bioinformatics/btu640
4. Hart SN et al (2013). RNA-seq sample sizes. https://doi.org/10.1089/cmb.2012.0283
5. RnaSeqSampleSize (Bioconductor). https://bioconductor.org/packages/release/bioc/html/RnaSeqSampleSize.html
6. Schmid KT et al (2021). scPower. https://doi.org/10.1038/s41467-021-26779-7
7. Crowell HL et al (2020). muscat. https://doi.org/10.1038/s41467-020-19894-4
8. Mathis A et al (2018). DeepLabCut. https://doi.org/10.1038/s41593-018-0209-y
9. Lazic SE (2010). Pseudoreplication. https://doi.org/10.1186/1471-2202-11-5
10. Percie du Sert N et al (2020). ARRIVE 2.0. https://doi.org/10.1371/journal.pbio.3000410
11. Percie du Sert N et al (2020). ARRIVE 2.0 explanation. https://doi.org/10.1371/journal.pbio.3000411
12. Riley RD et al (2020). Clinical prediction models. https://doi.org/10.1136/bmj.m441
13. MSstats, proteomics power. https://bioconductor.org/packages/release/bioc/html/MSstats.html
