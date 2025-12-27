#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import allel
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, to_tree, fcluster
from scipy.spatial.distance import squareform
from joblib import Parallel, delayed

# -------------------------------
# CONFIG
# -------------------------------
VCF_PATH   = "data/genomic_data/all_samples.vcf.gz"
META_PATH  = "data/genomic_data/sample_metadata.tsv"

OUT_CSV        = "data/genomic_data/ecotype_distance_matrix.csv"
OUT_PNG        = "data/genomic_data/ecotype_tree_upgma.png"
OUT_HEAT       = "data/genomic_data/ecotype_distance_heatmap.png"
OUT_NWK        = "data/genomic_data/ecotype_tree_upgma.nwk"
OUT_COPH       = "data/genomic_data/cophenetic_scatter.png"
OUT_BOOT_HIST  = "data/genomic_data/bootstrap_support_hist.png"
OUT_BOOT_TABLE = "data/genomic_data/ecotype_bootstrap_support.tsv"
OUT_SUMMARY    = "data/genomic_data/ecotype_tree_summary.tsv"
OUT_CLUSTER_DENSITY = "data/genomic_data/ecotype_cluster_within_between_density.png"
OUT_CLUSTER_BOX     = "data/genomic_data/ecotype_cluster_within_between_boxplot.png"
OUT_CLUSTER_STATS   = "data/genomic_data/ecotype_cluster_within_between_stats.tsv"

MAF_THRESHOLD      = 0.05
VAR_CALLRATE_MIN   = 0.80   # site-level call rate filter
THIN_WINDOW_BP     = 300    # 1 SNP per 300 bp window (proxy for 1 SNP per ddRAD locus/tag)

N_BOOTSTRAPS   = 500    # change to 100/1000 as desired
MAX_THREADS    = 32     # CPUs requested on the HPC
N_CLUSTERS     = 3      # number of color-coded clusters in dendrogram

ECOTYPES_OF_INTEREST = {"AK", "CF", "CH", "EC", "GK", "HL", "IM", "NN", "PB", "SG", "SL", "TL"}
RESTRICT_TO_ECOTYPES = True

# STRUCTURE cluster assignments from Stunetz et al 2021
STRUCTURE_CLUSTER_MAP = {
    "EC": "EC",
    "NC": "South",
    "VM": "South",
    "CC": "South",
    "EL": "South",
    "NN": "South",
    "GK": "South",
    "CF": "South",
    "ST": "South/North",
    "TB": "North/South",
    "CH": "North",
    "AT": "North",
    "TL": "North",
    "AK": "North",
    "SG": "North",
    "CP": "North",
    "PB": "North",
}

STRUCTURE_CLUSTER_COLORS = {
    "EC": "#f0f0f0ff",
    "South": "#ffe0e0",
    "North": "#e0f0ff",
    "South/North": "#f0e0ff",
    "North/South": "#f0e0ff",
}

STRUCTURE_CLUSTER_TEXT_COLORS = {
    "EC": "#707070",
    "South": "#cc6666",
    "North": "#4b79c4",
    "South/North": "#8b5bb5",
    "North/South": "#8b5bb5",
}

# -------------------------------
# Load VCF + metadata + MAF filter
# -------------------------------
def load_data_with_maf_filter():
    print("Loading metadata…")
    meta = pd.read_csv(META_PATH, sep="\t")

    print("Loading VCF…")
    callset = allel.read_vcf(
        VCF_PATH,
        fields=[
            "samples",
            "calldata/GT",
            "variants/CHROM",
            "variants/POS",
        ],
    )

    raw_samples = callset["samples"]
    samples = np.array([
        os.path.basename(s).replace(".sorted.bam", "")
        for s in raw_samples
    ])
    print("Samples:", samples)

    gt = allel.GenotypeArray(callset["calldata/GT"])
    total_snps = gt.shape[0]

    chrom = np.asarray(callset["variants/CHROM"]).astype(str)
    pos = np.asarray(callset["variants/POS"]).astype(np.int64)

    # Allele counts (handles missingness)
    ac = gt.count_alleles()
    called_alleles = ac.sum(axis=1)  # number of called alleles per variant across all samples

    # Biallelic filter: keep sites where only REF + 1 ALT are present
    if ac.shape[1] > 2:
        biallelic_mask = (ac[:, 2:].sum(axis=1) == 0)
    else:
        biallelic_mask = np.ones(gt.shape[0], dtype=bool)

    # MAF among called alleles
    p_alt = np.zeros(gt.shape[0], dtype=float)
    nonzero = called_alleles > 0
    p_alt[nonzero] = ac[nonzero, 1] / called_alleles[nonzero]
    maf = np.minimum(p_alt, 1.0 - p_alt)

    keep_mask = nonzero & biallelic_mask & (maf >= MAF_THRESHOLD)

    gt_filtered = gt[keep_mask]
    chrom_f = chrom[keep_mask]
    pos_f = pos[keep_mask]

    n_keep = int(np.sum(keep_mask))
    n_bial = int(np.sum(biallelic_mask))
    n_nonzero = int(np.sum(nonzero))

    print(f"Total variants in VCF: {total_snps:,}")
    print(f"Variants with >=1 called allele: {n_nonzero:,} ({n_nonzero/total_snps:.3f})")
    print(f"Biallelic variants (observed): {n_bial:,} ({n_bial/total_snps:.3f})")
    print(f"Kept after biallelic + MAF>={MAF_THRESHOLD}: {n_keep:,} ({n_keep/total_snps:.3f})")

    return meta, samples, gt_filtered, chrom_f, pos_f, total_snps, n_keep


def subset_samples_to_ecotypes(meta, samples, gt, allowed_ecotypes):
    """
    Filter to samples whose meta ecotype is in allowed_ecotypes.
    Returns filtered (meta, samples, gt) plus the set of ecotypes kept.
    """
    meta_idx = meta.set_index("sample_id")

    keep_sample_idx = []
    kept_ecotypes = set()

    for i, s in enumerate(samples):
        if s not in meta_idx.index:
            continue
        ec = meta_idx.loc[s, "ecotype"]
        if ec in allowed_ecotypes:
            keep_sample_idx.append(i)
            kept_ecotypes.add(ec)

    keep_sample_idx = np.array(keep_sample_idx, dtype=int)

    samples_f = samples[keep_sample_idx]
    gt_f = gt[:, keep_sample_idx, :]   # subset samples axis

    print(f"\nRestricting to ecotypes of interest:")
    print(f"  requested: {sorted(list(allowed_ecotypes))}")
    print(f"  kept ecotypes present in VCF+metadata: {sorted(list(kept_ecotypes))}")
    print(f"  kept samples: {len(samples_f)} / {len(samples)}")

    return meta, samples_f, gt_f, kept_ecotypes


# -------------------------------
# Site call-rate filter + 1 SNP per 300 bp window thinning
# -------------------------------
def filter_by_callrate_and_thin(gt, chrom, pos, min_callrate=0.80, window_bp=300):
    """
    1) Keep variants with site call rate >= min_callrate
    2) Within each chromosome, bin by (pos // window_bp) and keep 1 SNP per bin:
       the SNP with the highest call rate (ties broken by first encountered).
    """
    chrom = np.asarray(chrom).astype(str)
    pos = np.asarray(pos).astype(np.int64)

    # --- site call rate ---
    called = gt.is_called()                 # (n_variants, n_samples)
    site_callrate = called.mean(axis=1)     # fraction called per SNP

    keep1 = site_callrate >= float(min_callrate)
    gt = gt[keep1]
    chrom = chrom[keep1]
    pos = pos[keep1]
    site_callrate = site_callrate[keep1]

    print(f"\nSite call-rate filter >= {min_callrate:.2f}: kept {gt.shape[0]:,} variants")

    # --- thinning to 1 SNP per window_bp ---
    keep2 = np.zeros(gt.shape[0], dtype=bool)
    window_bp = int(window_bp)

    for c in np.unique(chrom):
        idx = np.where(chrom == c)[0]
        if idx.size == 0:
            continue

        order = np.argsort(pos[idx])
        idx = idx[order]

        bins = (pos[idx] // window_bp)
        uniq_bins = np.unique(bins)

        for b in uniq_bins:
            j = idx[bins == b]
            best = j[np.argmax(site_callrate[j])]
            keep2[best] = True

    gt = gt[keep2]
    chrom = chrom[keep2]
    pos = pos[keep2]

    print(f"Thinning to 1 SNP per {window_bp} bp window: kept {gt.shape[0]:,} variants")
    return gt, chrom, pos


# -------------------------------
# Allele Sharing Distance (ASD)
# -------------------------------
def compute_asd_matrix(samples, gt, n_jobs=1):
    """
    Compute sample-level ASD matrix in parallel.

    ASD here is mean(|alt_i - alt_j| / 2) over sites called in both samples.
    Missing genotypes are NOT imputed; they are excluded pairwise.
    """
    n = len(samples)
    D = np.zeros((n, n), dtype=float)

    # Convert once to alt-count array (n_snps x n_samples)
    g = gt.to_n_alt(fill=-1)

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
def collapse_to_ecotypes(meta, samples, dist, allowed_ecotypes=None):
    """
    Average sample-level distances into ecotype-level distances.
    """
    meta_idx = meta.set_index("sample_id")

    sample_ecotypes = []
    for s in samples:
        if s in meta_idx.index:
            sample_ecotypes.append(meta_idx.loc[s, "ecotype"])
        else:
            sample_ecotypes.append(None)

    ecotype_indices = {}
    for i, ec in enumerate(sample_ecotypes):
        if ec is None:
            continue
        if allowed_ecotypes is not None and ec not in allowed_ecotypes:
            continue
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
        return f"({left}:{node.dist:.6f},{right}:{node.dist:.6f})"

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
    """
    ref_clades = _get_clades(Z_ref, labels)
    clades_sorted = sorted(ref_clades, key=lambda c: (len(c), sorted(c)))

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
# Bootstrap ecotype tree support
# -------------------------------
def bootstrap_support(gt, samples, meta, ecotypes, n_boot=100, outer_n_jobs=1):
    """
    Bootstrap over SNPs to get replicate ecotype trees.
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

        if list(eco) != list(ecotypes):
            raise ValueError("Ecotype order mismatch between bootstrap replicate and reference.")

        dm = np.array(eco_dist, dtype=float, copy=True)
        dm = 0.5 * (dm + dm.T)
        np.fill_diagonal(dm, 0.0)
        condensed = squareform(dm)

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
# UPGMA tree + cophenetic validation + dendrogram
# -------------------------------
def build_upgma_tree(eco_dist, ecotypes, n_clusters=N_CLUSTERS):
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

    if n_clusters >= 2:
        color_threshold = Z[-(n_clusters - 1), 2] - 1e-8
    else:
        color_threshold = 0.0

    plt.figure(figsize=(8, 6))
    dendrogram(
        Z,
        labels=ecotypes,
        orientation="right",
        color_threshold=color_threshold,
        above_threshold_color="lightgrey",
        leaf_font_size=10,
    )
    plt.xlabel("Allele sharing distance")
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
# Bootstrap outputs
# -------------------------------
def save_bootstrap_outputs(ecotypes, clades, supports):
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

    plot_bootstrap_histogram(supports)


def plot_bootstrap_histogram(supports):
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
def save_summary(
    total_snps,
    n_snps_filtered,
    coph_coeff,
    n_ecotypes,
    n_boot,
    supports,
):
    summary = {
        "total_snps": [total_snps],
        "maf_threshold": [MAF_THRESHOLD],
        "site_callrate_min": [VAR_CALLRATE_MIN],
        "thin_window_bp": [THIN_WINDOW_BP],
        "snps_after_maf_filter": [n_snps_filtered],
        "n_ecotypes": [n_ecotypes],
        "cophenetic_correlation": [coph_coeff],
        "n_bootstraps": [n_boot],
        "mean_bootstrap_support": [float(np.mean(supports))],
        "median_bootstrap_support": [float(np.median(supports))],
        "min_bootstrap_support": [float(np.min(supports))],
        "max_bootstrap_support": [float(np.max(supports))],
    }

    df = pd.DataFrame(summary)
    df.to_csv(OUT_SUMMARY, sep="\t", index=False)
    print(f"Saved tree summary metrics → {OUT_SUMMARY}")


# -------------------------------
# Helpers for reuse
# -------------------------------
def load_ecotype_dist_from_csv():
    df = pd.read_csv(OUT_CSV, index_col=0)
    ecotypes = list(df.index)
    eco_dist = df.values
    return ecotypes, eco_dist


def load_bootstrap_supports_from_table():
    df = pd.read_csv(OUT_BOOT_TABLE, sep="\t")
    supports = df["bootstrap_support_percent"].values
    members = df["members"].tolist()
    clades = [frozenset(m.split(",")) for m in members]
    return clades, supports


# -------------------------------
# Dendrogram with STRUCTURE shading + bootstrap labels
# -------------------------------
def plot_dendrogram_with_bootstrap(
    Z,
    ecotypes,
    clades,
    supports,
    outfile,
    n_clusters=N_CLUSTERS,
    min_support=0,
    fontsize=8,
    equal_spacing=True,
):
    labels = list(ecotypes)
    clade_support = {c: s for c, s in zip(clades, supports)}

    tree, _ = to_tree(Z, rd=True)
    n_leaves = len(labels)
    node_clade = {}

    def traverse(node):
        if node.is_leaf():
            return frozenset([labels[node.id]])
        left = traverse(node.get_left())
        right = traverse(node.get_right())
        cl = left | right
        node_clade[node.id] = cl
        return cl

    traverse(tree)

    node_support = {
        node_id: clade_support[cl]
        for node_id, cl in node_clade.items()
        if cl in clade_support
    }

    if equal_spacing:
        Z_plot = Z.copy()
        uniq_heights = np.unique(Z_plot[:, 2])
        new_vals = np.arange(1, len(uniq_heights) + 1, dtype=float)
        height_map = dict(zip(uniq_heights, new_vals))
        Z_plot[:, 2] = np.array([height_map[h] for h in Z_plot[:, 2]], dtype=float)
    else:
        Z_plot = Z

    if n_clusters >= 2:
        color_threshold = Z_plot[-(n_clusters - 1), 2] - 1e-8
    else:
        color_threshold = 0.0

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ddata = dendrogram(
        Z_plot,
        labels=labels,
        orientation="right",
        color_threshold=color_threshold,
        above_threshold_color="lightgrey",
        leaf_font_size=10,
    )

    yticks = ax.get_yticks()
    ylabels = [t.get_text() for t in ax.get_yticklabels()]

    if len(yticks) > 1:
        dy = np.min(np.diff(np.sort(yticks))) * 0.5
    else:
        dy = 0.5

    from collections import defaultdict
    cluster_to_ys = defaultdict(list)
    for y, name in zip(yticks, ylabels):
        cluster = STRUCTURE_CLUSTER_MAP.get(name)
        if cluster is not None:
            cluster_to_ys[cluster].append(y)

    xmin, xmax = ax.get_xlim()
    for cluster, ys in cluster_to_ys.items():
        color = STRUCTURE_CLUSTER_COLORS.get(cluster)
        if color is None or len(ys) == 0:
            continue
        y_min = min(ys) - dy
        y_max = max(ys) + dy

        ax.axhspan(y_min, y_max, color=color, alpha=0.6, zorder=-2)

        if cluster in ("North/South", "South/North"):
            y_label = y_min
            va = "bottom"
        else:
            y_label = y_max
            va = "top"

        text_color = STRUCTURE_CLUSTER_TEXT_COLORS.get(cluster, "black")
        ax.text(
            xmax,
            y_label,
            cluster,
            ha="right",
            va=va,
            fontsize=8,
            color=text_color,
            zorder=5,
        )

    ax.set_yticks([])
    ax.set_yticklabels([])

    leaf_x = min(min(dc) for dc in ddata["dcoord"])
    for y, name in zip(yticks, ylabels):
        ax.text(leaf_x, y, name, ha="right", va="center", fontsize=10, zorder=5)

    for i, (icoord, dcoord) in enumerate(zip(ddata["icoord"], ddata["dcoord"])):
        node_id = i + n_leaves
        sup = node_support.get(node_id)
        if sup is None or sup < min_support:
            continue

        x = max(dcoord)
        y = 0.5 * (icoord[1] + icoord[2])

        plt.text(x + 0.1, y + 2.5, f"{sup:.0f}", va="center", ha="left",
                 fontsize=fontsize, color="#006400", zorder=10)

        asd = Z[i, 2]
        plt.text(x + 0.1, y + 6.0, f"{asd:.3f}", va="center", ha="left",
                 fontsize=fontsize, color="#1f3b99", zorder=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", left=False, right=False)

    plt.title("Ecotype UPGMA tree with bootstrap support")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Saved bootstrap-labeled dendrogram → {outfile}")


def compute_and_plot_within_between(eco_dist, ecotypes, Z, n_clusters=N_CLUSTERS):
    ecotypes = list(ecotypes)
    eco_dist = np.asarray(eco_dist)

    cluster_ids = fcluster(Z, t=n_clusters, criterion="maxclust")
    cluster_map = dict(zip(ecotypes, cluster_ids))
    print("Cluster assignments:")
    for e, c in cluster_map.items():
        print(f"  {e}: cluster {c}")

    within = []
    between = []
    n = len(ecotypes)
    for i in range(n):
        for j in range(i + 1, n):
            d = eco_dist[i, j]
            if cluster_ids[i] == cluster_ids[j]:
                within.append(d)
            else:
                between.append(d)

    within = np.array(within)
    between = np.array(between)

    def median_iqr(arr):
        med = np.median(arr)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        return med, q1, q3

    w_med, w_q1, w_q3 = median_iqr(within)
    b_med, b_q1, b_q3 = median_iqr(between)

    stats_df = pd.DataFrame([
        {"group": "within_cluster", "n": len(within), "median": w_med, "q1": w_q1, "q3": w_q3},
        {"group": "between_cluster", "n": len(between), "median": b_med, "q1": b_q1, "q3": b_q3},
    ])
    stats_df.to_csv(OUT_CLUSTER_STATS, sep="\t", index=False)
    print(f"Saved within/between summary stats → {OUT_CLUSTER_STATS}")

    df = pd.DataFrame({
        "distance": np.concatenate([within, between]),
        "group": (["within cluster"] * len(within)) + (["between clusters"] * len(between)),
    })

    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df, x="distance", hue="group", common_norm=False, fill=True, alpha=0.4)
    plt.xlabel("Genetic distance (ASD)")
    plt.ylabel("Density")
    plt.title("Within vs between cluster distances")
    plt.tight_layout()
    plt.savefig(OUT_CLUSTER_DENSITY, dpi=300)
    plt.close()
    print(f"Saved within/between density plot → {OUT_CLUSTER_DENSITY}")

    plt.figure(figsize=(5, 4))
    sns.boxplot(data=df, x="group", y="distance")
    plt.xlabel("")
    plt.ylabel("Genetic distance (ASD)")
    plt.title("Within vs between cluster distances")
    plt.tight_layout()
    plt.savefig(OUT_CLUSTER_BOX, dpi=300)
    plt.close()
    print(f"Saved within/between boxplot → {OUT_CLUSTER_BOX}")


# -------------------------------
# MAIN
# -------------------------------
def main():
    meta = None
    samples = None
    gt = None
    chrom = None
    pos = None
    total_snps = None
    n_snps_filtered = None

    # Reuse CSV only if not restricting ecotypes (to avoid mismatch)
    can_reuse_csv = os.path.exists(OUT_CSV)
    if RESTRICT_TO_ECOTYPES:
        can_reuse_csv = False

    # -------------------------------
    # STEP 1: Ecotype distance matrix
    # -------------------------------
    if can_reuse_csv:
        print(f"Found existing ecotype distance CSV → {OUT_CSV}")
        ecotypes, eco_dist = load_ecotype_dist_from_csv()
    else:
        print("Computing from VCF (fresh run).")
        meta, samples, gt, chrom, pos, total_snps, n_snps_filtered = load_data_with_maf_filter()

        # Restrict to ecotypes of interest (intersection with those present)
        allowed = set(ECOTYPES_OF_INTEREST)
        if RESTRICT_TO_ECOTYPES:
            meta, samples, gt, kept_ecotypes = subset_samples_to_ecotypes(meta, samples, gt, allowed)
            allowed = kept_ecotypes

        # Apply site call-rate filter + 1 SNP per 300 bp window thinning
        print(f"\nApplying site call-rate >= {VAR_CALLRATE_MIN:.2f} and thinning (1 SNP/{THIN_WINDOW_BP} bp)...")
        gt, chrom, pos = filter_by_callrate_and_thin(
            gt, chrom, pos,
            min_callrate=VAR_CALLRATE_MIN,
            window_bp=THIN_WINDOW_BP
        )

        print(f"\nComputing sample-level ASD distances with n_jobs={MAX_THREADS}...")
        D = compute_asd_matrix(samples, gt, n_jobs=MAX_THREADS)

        print("\nCollapsing into ecotypes...")
        ecotypes, eco_dist = collapse_to_ecotypes(meta, samples, D, allowed_ecotypes=allowed)

        pd.DataFrame(eco_dist, index=ecotypes, columns=ecotypes).to_csv(OUT_CSV)
        print(f"Saved ecotype distance CSV → {OUT_CSV}")

    # Always redo heatmap
    save_heatmap(ecotypes, eco_dist)
    print(f"Heatmap written → {OUT_HEAT}")

    # -------------------------------
    # STEP 2: UPGMA tree + cophenetic + dendrograms
    # -------------------------------
    print("Building UPGMA tree from ecotype distances...")
    Z, coph_coeff = build_upgma_tree(eco_dist, ecotypes, n_clusters=N_CLUSTERS)
    print("UPGMA tree saved.")
    print(f"Newick tree → {OUT_NWK}")
    print(f"Dendrogram PNG → {OUT_PNG}")
    print(f"Cophenetic scatter → {OUT_COPH}")

    compute_and_plot_within_between(eco_dist, ecotypes, Z, n_clusters=N_CLUSTERS)

    # -------------------------------
    # STEP 3: Bootstrap support + summary
    # -------------------------------
    have_boot_table = os.path.exists(OUT_BOOT_TABLE)
    have_summary = os.path.exists(OUT_SUMMARY)

    if have_boot_table and have_summary:
        print("Found existing bootstrap table and summary – skipping heavy bootstraps.")
        clades, supports = load_bootstrap_supports_from_table()
        plot_bootstrap_histogram(supports)

        plot_dendrogram_with_bootstrap(
            Z,
            ecotypes,
            clades,
            supports,
            outfile=OUT_PNG.replace(".png", "_boots.png"),
            n_clusters=N_CLUSTERS,
            min_support=50,
            equal_spacing=True,
        )
        print("Done (reused existing bootstrap results).")
        return

    # Ensure gt/meta/samples are loaded for bootstraps (and filtered the same way)
    if gt is None or meta is None or samples is None:
        meta, samples, gt, chrom, pos, total_snps, n_snps_filtered = load_data_with_maf_filter()

        allowed = set(ECOTYPES_OF_INTEREST)
        if RESTRICT_TO_ECOTYPES:
            meta, samples, gt, kept_ecotypes = subset_samples_to_ecotypes(meta, samples, gt, allowed)
            allowed = kept_ecotypes

        print(f"\nApplying site call-rate >= {VAR_CALLRATE_MIN:.2f} and thinning (1 SNP/{THIN_WINDOW_BP} bp)...")
        gt, chrom, pos = filter_by_callrate_and_thin(
            gt, chrom, pos,
            min_callrate=VAR_CALLRATE_MIN,
            window_bp=THIN_WINDOW_BP
        )

    print(f"\nRunning {N_BOOTSTRAPS} SNP bootstraps…")
    boot_trees = bootstrap_support(
        gt,
        samples,
        meta,
        ecotypes,
        n_boot=N_BOOTSTRAPS,
        outer_n_jobs=MAX_THREADS,
    )
    print(f"Computed {len(boot_trees)} bootstrap trees.")

    clades, supports = compute_bootstrap_support(Z, ecotypes, boot_trees)
    save_bootstrap_outputs(ecotypes, clades, supports)

    plot_dendrogram_with_bootstrap(
        Z,
        ecotypes,
        clades,
        supports,
        outfile=OUT_PNG.replace(".png", "_boots.png"),
        n_clusters=N_CLUSTERS,
        min_support=50,
        equal_spacing=True,
    )

    save_summary(
        total_snps=total_snps,
        n_snps_filtered=n_snps_filtered,
        coph_coeff=coph_coeff,
        n_ecotypes=len(ecotypes),
        n_boot=N_BOOTSTRAPS,
        supports=supports,
    )

    print("\nAll done!\n")


if __name__ == "__main__":
    main()
