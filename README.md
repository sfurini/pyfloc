## PyFLOC README
The Analysis of Flow Cytometry (FC) and Mass Cytometry (CyTOF) experiments by manual gating strategies is a time-consuming process, and it is also affected by inter-operator variability [Saeys, Y., Nat Rev Immuno.,16(7):449-62].  These shortcomings get more severe when more features are considered. Therefore, there is a great interest in computational tools for data analyses. AGACY is a novel software for automatic gating and clustering of FC and CyTOF experiments, with three main features: (1) it offers a unique framework for gating and data analysis, (2) it provides a robust estimate of the number of cell populations, and (3) thanks to a novel clustering algorithm it can identify rare populations with high precision.

![](https://github.com/sfurini/pyfloc/blob/master/docs/images/workflow.png)

# Workflow
After preprocessing, AGACY operates into a repeated sequence of two steps: clustering and sample selection. When clustering is performed in 2 dimensions, it is possible to choose one or more clusters, and to automatically trace a contour that selects the desired percentage of cells belonging to those clusters. The common analysis pipeline is to first run a series of iterations in 2-dimensional spaces to identify a subset of cells. Then, clustering in a multi-dimensional space is executed to identify cell populations and to compare different experimental conditions.

