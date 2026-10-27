"""Curated published primary-method references (not an exhaustive systematic review)."""
REFERENCES = [
  {'key':'ARRIVE','topic':'Experimental unit, animals','citation':'Percie du Sert N et al. (2020). The ARRIVE guidelines 2.0: updated guidelines for reporting animal research. PLOS Biology.','url':'https://doi.org/10.1371/journal.pbio.3000410','doi':'10.1371/journal.pbio.3000410'},
  {'key':'ARRIVE_EE','topic':'Sample-size rationale, animals','citation':'Percie du Sert N et al. (2020). Reporting animal research: explanation and elaboration for the ARRIVE guidelines 2.0. PLOS Biology.','url':'https://doi.org/10.1371/journal.pbio.3000411','doi':'10.1371/journal.pbio.3000411'},
  {'key':'BH','topic':'Multiple testing','citation':'Benjamini Y, Hochberg Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. JRSS B.','url':'https://doi.org/10.1111/j.2517-6161.1995.tb02031.x','doi':'10.1111/j.2517-6161.1995.tb02031.x'},
  {'key':'DESEQ2','topic':'Bulk RNA-seq differential expression','citation':'Love MI, Huber W, Anders S. (2014). Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. Genome Biology.','url':'https://doi.org/10.1186/s13059-014-0550-8','doi':'10.1186/s13059-014-0550-8'},
  {'key':'PROPER','topic':'RNA-seq sample size / simulation','citation':'Wu H, Wang C, Wu Z. (2015). PROPER: comprehensive power evaluation for differential expression using RNA-seq. Bioinformatics.','url':'https://doi.org/10.1093/bioinformatics/btu640','doi':'10.1093/bioinformatics/btu640'},
  {'key':'RNA_N','topic':'RNA-seq negative-binomial design','citation':'Hart SN et al. (2013). Calculating sample size estimates for RNA sequencing data. Journal of Computational Biology.','url':'https://doi.org/10.1089/cmb.2012.0283','doi':'10.1089/cmb.2012.0283'},
  {'key':'RNA_SS','topic':'Pilot-calibrated RNA-seq sample sizes','citation':'RnaSeqSampleSize. Bioconductor package, gene count and dispersion based design with FDR control.','url':'https://bioconductor.org/packages/release/bioc/html/RnaSeqSampleSize.html','doi':'10.18129/B9.bioc.RnaSeqSampleSize'},
  {'key':'SCPOWER','topic':'Multi-donor single cell design','citation':'Schmid KT et al. (2021). scPower accelerates and optimizes the design of multi-sample single cell transcriptomic studies. Nature Communications.','url':'https://doi.org/10.1038/s41467-021-26779-7','doi':'10.1038/s41467-021-26779-7'},
  {'key':'MUSCAT','topic':'Single-cell donor-level inference','citation':'Crowell HL et al. (2020). muscat detects subpopulation-specific state transitions from multi-sample multi-condition single-cell transcriptomics data. Nature Communications.','url':'https://doi.org/10.1038/s41467-020-19894-4','doi':'10.1038/s41467-020-19894-4'},
  {'key':'DLC','topic':'Behavior / pose extraction','citation':'Mathis A et al. (2018). DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nature Neuroscience.','url':'https://doi.org/10.1038/s41593-018-0209-y','doi':'10.1038/s41593-018-0209-y'},
  {'key':'PSEUDO','topic':'Pseudoreplication / animal behavior','citation':'Lazic SE. (2010). The problem of pseudoreplication in neuroscientific studies. BMC Neuroscience.','url':'https://doi.org/10.1186/1471-2202-11-5','doi':'10.1186/1471-2202-11-5'},
  {'key':'RILEY','topic':'Prediction model development / sample sizes','citation':'Riley RD et al. (2020). Calculating the sample size required for developing a clinical prediction model. BMJ.','url':'https://doi.org/10.1136/bmj.m441','doi':'10.1136/bmj.m441'},
  {'key':'MSSTATS','topic':'Proteomics variance based designs','citation':'MSstats user guide: designSampleSize based on a fitted pilot experiment.','url':'https://www.bioconductor.org/packages/release/bioc/html/MSstats.html','doi':''},
]


def get_refs(*keys):
    return [ref for ref in REFERENCES if ref['key'] in keys]
