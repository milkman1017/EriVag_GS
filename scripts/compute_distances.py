#!/usr/bin/env python3
import os

import numpy as np
import pandas as pd
import allel
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, to_tree
from scipy.spatial.distance import squareform
from Bio import SeqIO
from joblib import Parallel, delayed


# -------------------------------
# CONFIG
# -------------------------------
VCF_PATH = "data/genomic_data/all_samples.vcf.gz"
META_PATH = "data/genomic_data/sample_metadata.tsv"
REF_FASTA = "reference/eriophorum_vaginatum.fa"

OUT_CSV        = "data/genomic_data/ecotype_distance_matrix.csv"
OUT_PNG        = "data/genomic_data/ecotype_tree_upgma.png"
OUT_HEAT       = "data/genomic_data/ecotype_distance_heatmap.png"
OUT_NWK        = "data/genomic_data/ecotype_tree_upgma.nwk"
OUT_COPH       = "data/genomic_data/cophenetic_scatter.png"
OUT_BOOT_HIST  = "data/genomic_data/bootstrap_support_hist.png"
OUT_BOOT_TABLE = "data/genomic_data/ecotype_bootstrap_support.tsv"
OUT_SUMMARY    = "data/genomic_data/ecotype_tree_summary.tsv"

MAF_THRESHOLD  = 0.05
N_BOOTSTRAPS   = 500    # change to 100/1000 as desired
MAX_THREADS    = 32     # <-- set to the number of CPUs you requested on the HPC


# -------------------------------
# Load reference genome length (unused here but kept)
# -------------------------------
def load_reference_length(fasta_path):
    total_bp = 0
    for rec in SeqIO.parse(fasta_path, "fasta"):
        total_bp += len(rec.seq)
    return total_bp

# -------------------------------
# Load VCF + metadata + MAF filter
# -------------------------------
def load_data_with_maf_filter():
    print("Loading metadata…")
    meta = pd.read_csv(META_PATH, sep="\t")

    print("Loading VCF…")
    callset = allel.read_vcf(
        VCF_PATH,
        fields=["samples", "calldata/GT"]
    )

    raw_samples = callset["samples"]
    samples = np.array([
        os.path.basename(s).replace(".sorted.bam", "")
        for s in raw_samples
    ])

    print("Samples:", samples)

    gt = allel.GenotypeArray(callset["calldata/GT"])
    total_snps = gt.shape[0]

    # Compute MAF (bi-allelic assumption: allele 1)
    ac = gt.count_alleles()
    maf = ac[:, 1] / ac.sum(axis=1)

    keep_mask = maf >= MAF_THRESHOLD
    gt_filtered = gt[keep_mask]
    n_keep = int(np.sum(keep_mask))

    print(f"Total SNPs: {total_snps:,}")
    print(f"MAF>={MAF_THRESHOLD} SNPs: {n_keep:,} ({n_keep/total_snps:.3f})")

    return meta, samples, gt_filtered, total_snps, n_keep


# -------------------------------
# Allele Sharing Distance (ASD)
# -------------------------------
def compute_asd_matrix(samples, gt, n_jobs=1):
    """
    Compute sample-level ASD matrix in parallel.

    Parameters
    ----------
    samples : array-like (n_samples,)
    gt      : allel.GenotypeArray (n_snps, n_samples, ploidy)
    n_jobs  : int, number of workers

    Returns
    -------
    D : (n_samples, n_samples) numpy.ndarray
        Symmetric ASD distance matrix.
    """
    n = len(samples)
    D = np.zeros((n, n), dtype=float)

    # Convert once to alt-count array (n_snps x n_samples)
    g = gt.to_n_alt(fill=-1)

    # All i < j pairs
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def asd_worker(i, j):
        a = g[:, i]
        b = g[:, j]
        mask = (a >= 0) & (b >= 0)
        if mask.sum() == 0:
            return i, j, np.nan
        val = (np.abs(a[mask] - b[mask]) / 2.0).mean()
        return i, j, val

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(asd_worker)(i, j) for (i, j) in pairs
    )

    for i, j, val in results:
        D[i, j] = D[j, i] = val

    return D


# -------------------------------
# Collapse into ecotypes
# -------------------------------
def collapse_to_ecotypes(meta, samples, dist):
    """
    Average sample-level distances into ecotype-level distances.

    Returns
    -------
    ecotypes : list of ecotype names (sorted)
    eco_dist : (E, E) distance matrix (mean ASD between ecotypes)
    """
    meta_idx = meta.set_index("sample_id")

    sample_ecotypes = [
        meta_idx.loc[s, "ecotype"] if s in meta_idx.index else None
        for s in samples
    ]

    ecotype_indices = {}
    for i, ec in enumerate(sample_ecotypes):
        if ec is not None:
            ecotype_indices.setdefault(ec, []).append(i)

    ecotypes = sorted(ecotype_indices.keys())
    eco_dist = np.zeros((len(ecotypes), len(ecotypes)), dtype=float)

    for i, ec1 in enumerate(ecotypes):
        for j, ec2 in enumerate(ecotypes):
            sub = dist[np.ix_(ecotype_indices[ec1], ecotype_indices[ec2])]
            eco_dist[i, j] = np.nanmean(sub)

    return ecotypes, eco_dist


# -------------------------------
# Convert scipy linkage -> Newick
# -------------------------------
def scipy_cluster_to_newick(Z, labels):
    tree, _ = to_tree(Z, rd=True)

    def build_newick(node):
        if node.is_leaf():
            return labels[node.id]
        left = build_newick(node.get_left())
        right = build_newick(node.get_right())
        return f"({left}:{node.dist}, {right}:{node.dist})"

    return build_newick(tree) + ";"


# -------------------------------
# Clade utilities for bootstrap support
# -------------------------------
def _get_clades(Z, labels):
    """
    Return a list of clades (as frozensets of tip labels) from a linkage matrix.

    Excludes trivial clades of size 1 or all tips.
    """
    tree, _ = to_tree(Z, rd=True)
    labels = list(labels)
    n_tips = len(labels)

    def leaf_names(node):
        if node.is_leaf():
            return [labels[node.id]]
        return leaf_names(node.get_left()) + leaf_names(node.get_right())

    clades = []

    def traverse(node):
        if node.is_leaf():
            return
        leaves = leaf_names(node)
        if 1 < len(leaves) < n_tips:
            clades.append(frozenset(leaves))
        traverse(node.get_left())
        traverse(node.get_right())

    traverse(tree)
    return clades


def compute_bootstrap_support(Z_ref, labels, boot_trees):
    """
    Compute bootstrap support for each clade in the reference tree.

    Parameters
    ----------
    Z_ref      : linkage matrix of the reference tree
    labels     : list of tip labels
    boot_trees : list of linkage matrices from bootstrap replicates

    Returns
    -------
    clades_sorted : list[frozenset]
        Non-trivial clades from reference tree (sorted by size then name).
    supports      : np.ndarray
        Bootstrap support (%) for each clade.
    """
    ref_clades = _get_clades(Z_ref, labels)
    clades_sorted = sorted(
        ref_clades,
        key=lambda c: (len(c), sorted(c))
    )

    boot_clade_sets = []
    for Zb in boot_trees:
        boot_clade_sets.append(set(_get_clades(Zb, labels)))

    supports = []
    n_boot = len(boot_trees)

    for clade in clades_sorted:
        count = sum(clade in bcs for bcs in boot_clade_sets)
        supports.append(100.0 * count / n_boot)

    return clades_sorted, np.array(supports)


# -------------------------------
# Bootstrap ecotype distance support (parallel)
# -------------------------------
def bootstrap_support(gt, samples, meta, ecotypes, n_boot=100, outer_n_jobs=1):
    """
    Bootstrap over SNPs to get replicate ecotype trees.

    Parameters
    ----------
    gt       : allel.GenotypeArray (n_snps, n_samples, ploidy)
    samples  : array-like of sample names
    meta     : DataFrame with 'sample_id' and 'ecotype'
    ecotypes : list of ecotype labels from reference tree (sorted)
    n_boot   : int, number of bootstraps
    outer_n_jobs : int, workers for bootstrap loops

    Returns
    -------
    boot_trees : list of linkage matrices
    """
    n_snps = gt.shape[0]

    def one_bootstrap(seed):
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_snps, n_snps, replace=True)

        gt_boot = gt[idx]  # resampled SNPs

        # Sample-level ASD (inner parallel off to avoid oversubscription)
        D_boot = compute_asd_matrix(samples, gt_boot, n_jobs=1)

        # Collapse to ecotypes
        eco, eco_dist = collapse_to_ecotypes(meta, samples, D_boot)

        # Ensure ecotype order matches reference
        if list(eco) != list(ecotypes):
            raise ValueError(
                "Ecotype order mismatch between bootstrap replicate and reference."
            )

        # Proper distance matrix: symmetric, zero diagonal
        dm = np.array(eco_dist, dtype=float, copy=True)
        dm = 0.5 * (dm + dm.T)
        np.fill_diagonal(dm, 0.0)
        condensed = squareform(dm)

        # UPGMA on precomputed distances
        Z_boot = linkage(condensed, method="average")
        return Z_boot

    master_seed = np.random.SeedSequence()
    child_seeds = master_seed.spawn(n_boot)

    print(f"\nRunning {n_boot} SNP bootstraps (parallel, n_jobs={outer_n_jobs})...")
    boot_trees = Parallel(n_jobs=outer_n_jobs)(
        delayed(one_bootstrap)(int(cs.generate_state(1)[0])) for cs in child_seeds
    )

    return boot_trees


# -------------------------------
# UPGMA tree + cophenetic validation
# -------------------------------
def build_upgma_tree(eco_dist, ecotypes):
    """
    Build a UPGMA tree from ecotype distance matrix and compute cophenetic R.

    Returns
    -------
    Z          : linkage matrix
    coph_coeff : float
    """
    dm = np.array(eco_dist, dtype=float, copy=True)

    dm = 0.5 * (dm + dm.T)
    np.fill_diagonal(dm, 0.0)

    condensed = squareform(dm)

    Z = linkage(condensed, method="average")

    coph_coeff, coph_dists = cophenet(Z, condensed)
    print("\nCophenetic correlation coefficient:", coph_coeff)

    plt.figure(figsize=(6, 6))
    plt.scatter(condensed, coph_dists, s=10)
    plt.xlabel("Original ecotype distance (ASD)")
    plt.ylabel("Cophenetic distance (tree)")
    plt.title(f"Cophenetic correlation: {coph_coeff:.3f}")
    plt.tight_layout()
    plt.savefig(OUT_COPH, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=ecotypes, orientation="right")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()

    newick = scipy_cluster_to_newick(Z, ecotypes)
    with open(OUT_NWK, "w") as f:
        f.write(newick)

    return Z, coph_coeff


# -------------------------------
# Heatmap plot
# -------------------------------
def save_heatmap(ecotypes, eco_dist):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        eco_dist,
        xticklabels=ecotypes,
        yticklabels=ecotypes,
        cmap="viridis",
        square=True
    )
    plt.title("Ecotype Genetic Distance (ASD)")
    plt.tight_layout()
    plt.savefig(OUT_HEAT, dpi=300)
    plt.close()


# -------------------------------
# Plot & save bootstrap support
# -------------------------------
def save_bootstrap_outputs(ecotypes, clades, supports):
    """
    Save:
      - TSV with each clade and its bootstrap support
      - Histogram of bootstrap supports
    """
    rows = []
    for clade, sup in zip(clades, supports):
        members = sorted(list(clade))
        rows.append({
            "clade_size": len(members),
            "members": ",".join(members),
            "bootstrap_support_percent": sup
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_BOOT_TABLE, sep="\t", index=False)
    print(f"Saved bootstrap support table → {OUT_BOOT_TABLE}")

    plt.figure(figsize=(6, 4))
    plt.hist(supports, bins=np.arange(0, 110, 10), edgecolor="black")
    plt.xlabel("Bootstrap support (%)")
    plt.ylabel("Number of clades")
    plt.title("Distribution of bootstrap support (ecotype tree)")
    plt.tight_layout()
    plt.savefig(OUT_BOOT_HIST, dpi=300)
    plt.close()
    print(f"Saved bootstrap support histogram → {OUT_BOOT_HIST}")


# -------------------------------
# Save summary table (for paper)
# -------------------------------
def save_summary(total_snps, n_snps_filtered, coph_coeff, n_ecotypes, n_boot,
                 supports):
    summary = {
        "total_snps": [total_snps],
        "maf_threshold": [MAF_THRESHOLD],
        "snps_after_maf_filter": [n_snps_filtered],
        "n_ecotypes": [n_ecotypes],
        "cophenetic_correlation": [coph_coeff],
        "n_bootstraps": [n_boot],
        "mean_bootstrap_support": [float(np.mean(supports))],
        "median_bootstrap_support": [float(np.median(supports))],
        "min_bootstrap_support": [float(np.min(supports))],
        "max_bootstrap_support": [float(np.max(supports))]
    }

    df = pd.DataFrame(summary)
    df.to_csv(OUT_SUMMARY, sep="\t", index=False)
    print(f"Saved tree summary metrics → {OUT_SUMMARY}")


# -------------------------------
# MAIN
# -------------------------------
def main():
    # Load and filter data
    meta, samples, gt, total_snps, n_snps_filtered = load_data_with_maf_filter()

    # Compute sample-level ASD (parallel, bounded by MAX_THREADS)
    print(f"\nComputing sample-level distances with n_jobs={MAX_THREADS}...")
    D = compute_asd_matrix(samples, gt, n_jobs=MAX_THREADS)

    # Collapse to ecotypes
    print("\nCollapsing into ecotypes...")
    ecotypes, eco_dist = collapse_to_ecotypes(meta, samples, D)

    pd.DataFrame(eco_dist, index=ecotypes, columns=ecotypes).to_csv(OUT_CSV)
    print(f"Saved ecotype distance CSV → {OUT_CSV}")

    save_heatmap(ecotypes, eco_dist)
    print(f"Saved ecotype distance heatmap → {OUT_HEAT}")

    Z, coph_coeff = build_upgma_tree(eco_dist, ecotypes)
    print("UPGMA tree saved.")
    print(f"Newick tree → {OUT_NWK}")
    print(f"Dendrogram PNG → {OUT_PNG}")
    print(f"Cophenetic scatter → {OUT_COPH}")

    # Bootstrap SNPs to get replicate trees (bounded by MAX_THREADS)
    boot_trees = bootstrap_support(
        gt,
        samples,
        meta,
        ecotypes,
        n_boot=N_BOOTSTRAPS,
        outer_n_jobs=MAX_THREADS
    )
    print(f"Computed {len(boot_trees)} bootstrap trees.")

    clades, supports = compute_bootstrap_support(Z, ecotypes, boot_trees)
    save_bootstrap_outputs(ecotypes, clades, supports)

    save_summary(
        total_snps=total_snps,
        n_snps_filtered=n_snps_filtered,
        coph_coeff=coph_coeff,
        n_ecotypes=len(ecotypes),
        n_boot=N_BOOTSTRAPS,
        supports=supports
    )

    print("\nAll done!\n")


if __name__ == "__main__":
    main()
