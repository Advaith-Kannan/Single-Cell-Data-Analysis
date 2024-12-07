# %% [markdown]
# # Assignment-1: Computational Genomics
# ## Task 1.1
# A dataset "dataset.h5ad" has been provided, which contains a single-cell gene expression matrix. The following are the tasks to be performed on this dataset. 
# 1. Quality control, normalization, feature selection and PCA.
# 2. Following this, data is to be clustered using the Leiden Clustering Algorithm. 
# 3. Differential expression analysis is to be performed upon each cluster to identify the marker genes.  
# 4. Based on the marker genes identified, the clusters obtained are to be annotated. 

# %% [markdown]
# 
# 
# ### Loading the Dataset

# %%
pip install scanpy

# %%
import scanpy as sc 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# %%
matrix = sc.read('/kaggle/input/sequencingdata/dataset.h5ad')
matrix

# %% [markdown]
# What does this output mean? 
# - n_obs refers to the number of cells. Hence, here we have 10,727 cells. 
# - n_vars refers to the number of genes. This matrix contains details about 12,303 genes. 
# 
# Inspecting the AnnData object. 

# %%
matrix.var_names

# %%
matrix.obs_names

# %% [markdown]
# Adding the gene IDs provided.

# %%
gene_IDs = pd.read_csv('/kaggle/input/sequencingdata/gene_names.csv')
gene_IDs

# %%
matrix.var_names = gene_IDs['0'].values
matrix.var_names

# %% [markdown]
# ### Quality Control
# Finding the mitochondrial, ribosomal and hemoglobin genes. 

# %%
# mitochondrial genes
matrix.var["mt"] = matrix.var_names.str.startswith("MT-")
# ribosomal genes
matrix.var["ribo"] = matrix.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin genes
matrix.var["hb"] = matrix.var_names.str.contains("^HB[^(P)]")
sc.pp.calculate_qc_metrics(
    matrix, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
)

# %% [markdown]
# The function above, adds the qc metrics to our AnnData object. The log1p=True returns the logarithm of the results (log(1+x)), which accounts for skewed distrubutions. 

# %%
sc.pl.violin(
    matrix,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
)

# %% [markdown]
# It is clear from the above violin plots that there are no mitochondrial genes expressed. 

# %%
sc.pl.scatter(matrix, "total_counts", "n_genes_by_counts")

# %% [markdown]
# To choose apt quality control thresholds, it may be helpful to draw histograms. 

# %%
# Plot a histogram of total counts
plt.hist(matrix.obs['total_counts'], bins=100, color='skyblue', edgecolor='black')
plt.title('Total Counts per Cell')
plt.xlabel('Total Counts')
plt.ylabel('Number of Cells')
plt.show()

# %% [markdown]
# There seems to be a noise peak in the above plot below the value of 1400 total counts. 

# %%
# Plot a histogram of number of genes per barcode
plt.hist(matrix.obs['n_genes_by_counts'], bins=100, color='skyblue', edgecolor='black')
plt.title('Number of Unique Genes Expressed Per Cell')
plt.xlabel('Number of Unique Genes')
plt.ylabel('Number of Cells')
plt.show()

# %% [markdown]
# In this graph there is no apparent noise peak on the lower end. 

# %%
TotalCounts = matrix.obs['total_counts'].values
sorted_counts = np.sort(TotalCounts)[::-1]
ranks = np.arange(1, len(sorted_counts) + 1)
plt.figure(figsize=(8, 6))
plt.plot(ranks, sorted_counts, marker='o', linestyle='none', color='blue', markersize=1)
plt.yscale('log')
plt.xlabel('Rank of Cells (High to Low)')
ax = plt.gca()
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs='auto'))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=10))
ax.yaxis.set_major_formatter(ticker.LogFormatter(base=10.0))
plt.ylabel('Total Counts (log scale)')
plt.title('Count Depth Distribution (Log-Log Plot)')
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.show()

# %% [markdown]
# In the above graph, the "Elbow" is around 1600 total counts.

# %% [markdown]
# ### Doublet Detection
# Making use of Scrublet - the nearest neighbour based doublet detection algorithm.

# %%
sc.pp.scrublet(matrix)
np.sum(matrix.obs['predicted_doublet'])

# %% [markdown]
# Hence, 161 doublets have been detected.

# %% [markdown]
# ### Normalization
# 

# %%
#Saving the raw counts
matrix.layers["counts"] = matrix.X.copy()
# Normalizing to median total counts and transforming it to log(1+x)
sc.pp.normalize_total(matrix)
sc.pp.log1p(matrix)

# %% [markdown]
# ### Feature Selection
# This step is to reduce the dimensions of the matrix such that only the most informative genes are retained. Scanpy automatically annotates the highly variable genes using variance to mean ratios. Usually, erring towards the removal of a larger number of highly variable genes often produces the best results. Hence, here we choose the top 2000 most variable genes.   

# %%
sc.pp.highly_variable_genes(matrix, n_top_genes=2000)
sc.pl.highly_variable_genes(matrix)

# %% [markdown]
# To be even safer, let us view the top 5000 highly variable genes. 

# %%
sc.pp.highly_variable_genes(matrix, n_top_genes=5000)
sc.pl.highly_variable_genes(matrix)

# %%
matrix

# %% [markdown]
# As visible, the 'highly_variable' key has now been added to var. 

# %% [markdown]
# ### Dimensionality Reduction
# Performing Principal Component Analysis (PCA).

# %%
matrix = matrix.copy()
sc.tl.pca(matrix)
sc.pl.pca_variance_ratio(matrix, n_pcs=50, log=True)


# %% [markdown]
# The "elbow" of the above curve is around the region of 10-11 principal components. Since there are no pronounced disadvantages of overestimating the principal components, we can fix the value at 30 principal components.

# %%
sc.pp.pca(matrix, n_comps = 30)

# %% [markdown]
# ### Constructing the Nearest Neighbours Graph
# 

# %%
sc.pp.neighbors(matrix)
sc.tl.umap(matrix)
sc.pl.umap(matrix, size = 2,)

# %% [markdown]
# ### Visualizing the QC Metrics

# %%
sc.pl.umap(
    matrix,
    color=["predicted_doublet", "doublet_score"],
    # increase horizontal space between panels
    wspace=0.5,
    size=3,
)

# %%
sc.pl.umap(
    matrix,
    color=["log1p_total_counts", "log1p_n_genes_by_counts"],
    wspace=0.5,
    ncols=2,
)

# %%
sc.pl.umap(
    matrix,
    color=["log1p_total_counts", "log1p_n_genes_by_counts"],
    wspace=0.5,
    ncols=2,
)

# %% [markdown]
# Let us now find the highly variable genes for clustering.

# %%
#5000
sc.pp.highly_variable_genes(matrix, n_top_genes=5000)
matrix_5000 = matrix[:, matrix.var["highly_variable"]]
sc.pp.neighbors(matrix_5000)
sc.tl.umap(matrix_5000)
sc.pl.umap(matrix_5000, color = ['total_counts'], size = 2,)


# %%
matrix_5000

# %%
matrix = matrix_5000

# %% [markdown]
# # Clustering
# Making use of leiden clustering. Let us observe how the clusters form at multiple different resolutions. 

# %%
for res in [0.05, 0.1, 0.20, 0.25, 0.30, 1.0, 2.0]:
    sc.tl.leiden(
        matrix, key_added=f"leiden_res_{res:4.2f}", resolution=res, flavor="igraph"
    )

# %%
sc.pl.umap(
    matrix,
    color=["leiden_res_0.05", "leiden_res_0.10", "leiden_res_0.20", "leiden_res_0.25", "leiden_res_0.30", "leiden_res_1.00", "leiden_res_2.00" ],
    legend_loc="on data",
)

# %% [markdown]
# All clusters above 0.2 seem somewhat overclustered. Let us work further with the 0.20 resolution Leiden clusters.

# %%
#Obtaining Output Data
leiden_out = matrix.obs['leiden_res_0.20']
leiden_df = pd.DataFrame(leiden_out)
leiden_df.to_csv('leiden-0.20.csv', index=True)


# %%
# Obtain cluster-specific differentially expressed genes
sc.tl.rank_genes_groups(matrix, groupby="leiden_res_0.20", method="wilcoxon")
sc.pl.rank_genes_groups_dotplot(
    matrix, groupby="leiden_res_0.20",standard_scale="var", n_genes=5
)

# %% [markdown]
# Searching the differentially expressed genes with PanglaoDB, we can manually annotate the clusters produced.


