#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import allel
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed


# -------------------------------
# CONFIG
# -------------------------------
VCF_PATH    = "data/genomic_data/all_samples.vcf.gz"
META_PATH   = "data/genomic_data/sample_metadata.tsv"
PHENO_PATH  = "data/genomic_data/ecotypes_PStraits.csv"

OUT_DIR     = "data/genomic_data/gwas_results"
os.makedirs(OUT_DIR, exist_ok=True)

TRAITS      = ["vcmax", "jmax", "dark_resp",
               "light_comp_point", "pgmax", "quantum_yield"]

MAF_THRESHOLD    = 0.05
MISS_THRESHOLD   = 0.10
N_PCS            = 3
MAX_THREADS      = 16
REGION_WINDOW_BP = 50_000
TOP_SNPS_LOO     = 5

# PCA should be computed on LD-pruned SNPs
LD_PRUNE_R2_THRESHOLD = 0.2     # keep roughly "unlinked" SNPs
LD_PRUNE_SIZE         = 200     # window size in number of variants (index-based)
LD_PRUNE_STEP         = 20      # step in number of variants

# LD plotting/validation (subsample to keep it fast)
LD_DECAY_MAX_DIST_BP   = 200_000
LD_DECAY_N_PAIRS       = 50_000
LD_DECAY_N_BINS        = 50
LD_HEATMAP_N_SNPS      = 200
LD_VALIDATE_N_PAIRS    = 50_000
LD_VALIDATE_MAX_DIST_BP = 50_000

META_SAMPLE_COL  = "sample_id"
META_ECO_COL     = "ecotype"
PHENO_ECO_COL    = "eco"


# -------------------------------
# Utilities
# -------------------------------
def _masked_gn_from_gt(gt):
    """Return masked genotype alt-count array (n_variants x n_samples) with missing masked."""
    g_alt = gt.to_n_alt()  # -1 for missing
    return np.ma.masked_equal(g_alt, -1)

def _get_locate_unlinked():
    """Handle different scikit-allel placements of locate_unlinked."""
    if hasattr(allel, "locate_unlinked"):
        return allel.locate_unlinked
    if hasattr(allel, "stats") and hasattr(allel.stats, "ld") and hasattr(allel.stats.ld, "locate_unlinked"):
        return allel.stats.ld.locate_unlinked
    raise AttributeError("Could not find locate_unlinked in scikit-allel.")

def _rogers_huff_r_pair(g0, g1):
    """
    Pairwise LD correlation r between two variants using alt-count genotypes (0/1/2).
    Handles masked/missing by mean-imputation per variant.
    """
    # Convert to float arrays with NaN for missing
    if np.ma.isMaskedArray(g0):
        a = g0.filled(np.nan).astype(float)
    else:
        a = np.array(g0, dtype=float)

    if np.ma.isMaskedArray(g1):
        b = g1.filled(np.nan).astype(float)
    else:
        b = np.array(g1, dtype=float)

    # Mean-impute NaNs
    if np.any(~np.isfinite(a)):
        am = np.nanmean(a)
        a = np.where(np.isfinite(a), a, am)
    if np.any(~np.isfinite(b)):
        bm = np.nanmean(b)
        b = np.where(np.isfinite(b), b, bm)

    # Pearson correlation
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = np.sqrt(np.dot(a0, a0) * np.dot(b0, b0))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a0, b0) / denom)



# -------------------------------
# Load VCF + metadata
# -------------------------------
def load_vcf_and_meta():
    print("Loading metadata…")
    meta = pd.read_csv(META_PATH, sep="\t")

    print("Loading VCF…")
    callset = allel.read_vcf(
        VCF_PATH,
        fields=["samples", "calldata/GT", "variants/CHROM", "variants/POS"]
    )

    raw_samples = callset["samples"]
    vcf_samples = np.array([
        os.path.basename(s).replace(".sorted.bam", "")
        for s in raw_samples
    ])

    gt = allel.GenotypeArray(callset["calldata/GT"])
    chrom = np.array(callset["variants/CHROM"])
    pos   = np.array(callset["variants/POS"])

    print(f"N samples in VCF: {len(vcf_samples)}")
    print(f"N SNPs in VCF: {gt.shape[0]:,}")

    return meta, vcf_samples, gt, chrom, pos


# -------------------------------
# Load & align phenotypes
# -------------------------------
def load_and_merge_phenotypes(meta, vcf_samples):
    print("Loading phenotypes…")
    pheno = pd.read_csv(PHENO_PATH)

    if PHENO_ECO_COL not in pheno.columns:
        raise KeyError(
            f"Phenotype file is missing ecotype column '{PHENO_ECO_COL}'. "
            f"Available columns: {list(pheno.columns)}"
        )
    for col in (META_ECO_COL, META_SAMPLE_COL):
        if col not in meta.columns:
            raise KeyError(
                f"Metadata file is missing column '{col}'. "
                f"Available columns: {list(meta.columns)}"
            )

    missing_traits = [t for t in TRAITS if t not in pheno.columns]
    if missing_traits:
        raise KeyError(f"The following TRAITS are missing in phenotype file: {missing_traits}")

    # ecotype means
    pheno_eco = (
        pheno
        .groupby(PHENO_ECO_COL)[TRAITS]
        .mean()
        .reset_index()
    )

    merged = meta.merge(
        pheno_eco,
        left_on=META_ECO_COL,
        right_on=PHENO_ECO_COL,
        how="inner"
    )

    merged = merged[merged[META_SAMPLE_COL].isin(vcf_samples)].copy()

    # reorder to VCF sample order
    order_idx = []
    for sid in merged[META_SAMPLE_COL]:
        idx = np.where(vcf_samples == sid)[0]
        if len(idx) == 0:
            raise ValueError(f"Sample {sid} in phenotypes not found in VCF samples.")
        order_idx.append(idx[0])
    order_idx = np.array(order_idx)

    merged = merged.iloc[np.argsort(order_idx)].reset_index(drop=True)
    vcf_sample_order = vcf_samples[np.sort(order_idx)]

    print(f"N samples with both genotypes and ecotype-level phenotypes: {merged.shape[0]}")
    return merged, vcf_sample_order, pheno_eco


# -------------------------------
# SNP filtering
# -------------------------------
def filter_snps(gt, maf_threshold=0.05, miss_threshold=0.10):
    print("Filtering SNPs by MAF and missingness...")

    g_alt = gt.to_n_alt()  # n_snps x n_samples

    missing = (g_alt < 0)
    miss_rate = missing.mean(axis=1)

    g_alt_nonmiss = np.where(missing, np.nan, g_alt)
    ac_ref = np.nansum(2 - g_alt_nonmiss, axis=1)
    ac_alt = np.nansum(g_alt_nonmiss, axis=1)
    ac_tot = ac_ref + ac_alt

    maf = ac_alt / ac_tot
    maf = np.minimum(maf, 1 - maf)

    maf_mask  = maf >= maf_threshold
    miss_mask = miss_rate <= miss_threshold
    keep = maf_mask & miss_mask

    print(f"Total SNPs: {gt.shape[0]:,}")
    print(f"SNPs after filters: {np.sum(keep):,} "
          f"(MAF ≥ {maf_threshold}, missing ≤ {miss_threshold})")

    return keep, maf, miss_rate


# -------------------------------
# LD pruning (for PCA)
# -------------------------------
def ld_prune_by_chrom(gt_filt, chrom_filt,
                      size=LD_PRUNE_SIZE, step=LD_PRUNE_STEP,
                      threshold=LD_PRUNE_R2_THRESHOLD):
    """
    LD prune SNPs within each chromosome/contig separately using locate_unlinked.
    This is index-window LD pruning (not bp-window), which is standard for PCA prep.
    """
    print("LD pruning SNPs for PCA...")
    locate_unlinked = _get_locate_unlinked()

    gn_all = _masked_gn_from_gt(gt_filt)  # n_variants x n_samples (masked)
    keep_ld = np.zeros(gt_filt.shape[0], dtype=bool)

    chroms = pd.unique(pd.Series(chrom_filt.astype(str)))
    for c in chroms:
        idx = np.where(chrom_filt.astype(str) == c)[0]
        if idx.size == 0:
            continue
        gn_chr = gn_all[idx, :]

        # locate_unlinked returns True for kept (unlinked) variants
        kept_chr = locate_unlinked(gn_chr, size=size, step=step, threshold=threshold)
        keep_ld[idx] = kept_chr

        print(f"  {c}: {idx.size:,} → {int(np.sum(kept_chr)):,} kept (r^2<thr ~ {threshold})")

    print(f"LD-pruned SNPs for PCA: {int(np.sum(keep_ld)):,} / {gt_filt.shape[0]:,}")
    out = os.path.join(OUT_DIR, "ld_prune_summary.tsv")
    pd.DataFrame({
        "ld_prune_r2_threshold": [threshold],
        "ld_prune_size_variants": [size],
        "ld_prune_step_variants": [step],
        "n_snps_input": [int(gt_filt.shape[0])],
        "n_snps_kept": [int(np.sum(keep_ld))],
    }).to_csv(out, sep="\t", index=False)
    print(f"Saved LD prune summary → {out}")

    return keep_ld


# -------------------------------
# LD plots + validation
# -------------------------------
def plot_ld_heatmap(gt_filt, chrom_filt, pos_filt, out_png,
                    target_chrom=None, n_snps=LD_HEATMAP_N_SNPS):
    """
    Plot an LD (r^2) heatmap for a small region on one chromosome.
    Robust: computes r^2 matrix via numpy corrcoef (no dependence on allel LD API).
    """
    gn_all = _masked_gn_from_gt(gt_filt)

    chroms = chrom_filt.astype(str)
    pos = pos_filt.astype(int)

    # choose chromosome/contig with most SNPs if not provided
    if target_chrom is None:
        uniq, counts = np.unique(chroms, return_counts=True)
        target_chrom = uniq[np.argmax(counts)]

    idx = np.where(chroms == str(target_chrom))[0]
    if idx.size < 10:
        print("Not enough SNPs on chosen chrom for LD heatmap; skipping.")
        return

    # ensure sorted by position
    idx = idx[np.argsort(pos[idx])]

    # take evenly spaced SNPs across this contig
    take = min(n_snps, idx.size)
    pick = idx[np.linspace(0, idx.size - 1, take).astype(int)]

    # variants x samples, with NaN for missing
    g = gn_all[pick, :].filled(np.nan).astype(float)

    # mean-impute missing per variant
    m = np.nanmean(g, axis=1)
    g = np.where(np.isfinite(g), g, m[:, None])

    # corrcoef expects observations in columns if rowvar=False
    # Here: samples are columns, variants are rows => rowvar=True is fine,
    # but easiest is transpose and use rowvar=False:
    r = np.corrcoef(g, rowvar=True)   # (take x take)
    r2 = r * r

    plt.figure(figsize=(6, 5))
    plt.imshow(r2, origin="lower", aspect="auto")
    plt.colorbar(label="r²")
    plt.title(f"LD heatmap (r²) – {target_chrom} (n={take} SNPs)")
    plt.xlabel("SNP index (subsampled)")
    plt.ylabel("SNP index (subsampled)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved LD heatmap → {out_png}")



def plot_ld_decay(gt_filt, chrom_filt, pos_filt, out_png,
                  max_dist_bp=LD_DECAY_MAX_DIST_BP,
                  n_pairs=LD_DECAY_N_PAIRS,
                  n_bins=LD_DECAY_N_BINS,
                  seed=1):
    """
    Sample random SNP pairs within max_dist_bp on the same chrom and plot mean r² vs distance.
    """
    rng = np.random.default_rng(seed)
    gn_all = _masked_gn_from_gt(gt_filt)

    chroms = chrom_filt.astype(str)
    pos = pos_filt.astype(int)

    # build per-chrom index lists for sampling
    chrom_to_idx = {}
    for c in np.unique(chroms):
        idx = np.where(chroms == c)[0]
        if idx.size >= 2:
            chrom_to_idx[c] = idx

    if len(chrom_to_idx) == 0:
        print("No chromosomes with >=2 SNPs for LD decay; skipping.")
        return

    dists = []
    r2s = []

    chrom_keys = list(chrom_to_idx.keys())

    for _ in range(n_pairs):
        c = chrom_keys[rng.integers(0, len(chrom_keys))]
        idx = chrom_to_idx[c]

        i = idx[rng.integers(0, idx.size)]
        # choose j within bp window
        jmax = np.searchsorted(pos[idx], pos[i] + max_dist_bp, side="right") - 1
        if jmax <= 0:
            continue
        # jmax is in local idx coordinates
        iloc = np.searchsorted(pos[idx], pos[i], side="left")
        if jmax <= iloc:
            continue
        jloc = rng.integers(iloc + 1, jmax + 1)
        j = idx[jloc]

        dist = abs(pos[j] - pos[i])
        if dist <= 0 or dist > max_dist_bp:
            continue

        r = _rogers_huff_r_pair(gn_all[i, :], gn_all[j, :])
        dists.append(dist)
        r2s.append(r * r)

    if len(dists) < 100:
        print("Too few LD pairs sampled for decay plot; skipping.")
        return

    dists = np.array(dists)
    r2s = np.array(r2s)

    bins = np.linspace(0, max_dist_bp, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_r2 = np.full(n_bins, np.nan)
    n_in_bin = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        m = (dists >= bins[b]) & (dists < bins[b + 1])
        n_in_bin[b] = int(np.sum(m))
        if n_in_bin[b] > 0:
            mean_r2[b] = float(np.mean(r2s[m]))

    plt.figure(figsize=(6, 4))
    plt.plot(bin_centers, mean_r2, marker="o", linewidth=1)
    plt.xlabel("Distance (bp)")
    plt.ylabel("Mean r²")
    plt.title(f"LD decay (sampled pairs, max {max_dist_bp:,} bp)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved LD decay plot → {out_png}")


def validate_ld_pruning(gt_filt, chrom_filt, pos_filt, keep_ld_mask,
                        out_prefix,
                        max_dist_bp=LD_VALIDATE_MAX_DIST_BP,
                        n_pairs=LD_VALIDATE_N_PAIRS,
                        r2_threshold=LD_PRUNE_R2_THRESHOLD,
                        seed=2):
    """
    Validate pruning by comparing r² distribution (within max_dist_bp) before vs after.
    Writes TSV + histogram plot.
    """
    rng = np.random.default_rng(seed)

    def sample_r2(gt_obj, chrom_arr, pos_arr, n_pairs):
        gn_all = _masked_gn_from_gt(gt_obj)
        chroms = chrom_arr.astype(str)
        pos = pos_arr.astype(int)

        chrom_to_idx = {}
        for c in np.unique(chroms):
            idx = np.where(chroms == c)[0]
            if idx.size >= 2:
                chrom_to_idx[c] = idx
        if len(chrom_to_idx) == 0:
            return np.array([])

        chrom_keys = list(chrom_to_idx.keys())
        r2s = []

        for _ in range(n_pairs):
            c = chrom_keys[rng.integers(0, len(chrom_keys))]
            idx = chrom_to_idx[c]

            i = idx[rng.integers(0, idx.size)]
            # j within bp window
            jmax = np.searchsorted(pos[idx], pos[i] + max_dist_bp, side="right") - 1
            if jmax <= 0:
                continue
            iloc = np.searchsorted(pos[idx], pos[i], side="left")
            if jmax <= iloc:
                continue
            jloc = rng.integers(iloc + 1, jmax + 1)
            j = idx[jloc]

            dist = abs(pos[j] - pos[i])
            if dist <= 0 or dist > max_dist_bp:
                continue

            r = _rogers_huff_r_pair(gn_all[i, :], gn_all[j, :])
            r2s.append(r * r)

        return np.array(r2s, dtype=float)

    # before pruning
    r2_before = sample_r2(gt_filt, chrom_filt, pos_filt, n_pairs)

    # after pruning
    gt_pruned = gt_filt[keep_ld_mask]
    chrom_pruned = chrom_filt[keep_ld_mask]
    pos_pruned = pos_filt[keep_ld_mask]
    r2_after = sample_r2(gt_pruned, chrom_pruned, pos_pruned, n_pairs)

    def summarize(x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"n": 0}
        return {
            "n": int(x.size),
            "mean_r2": float(np.mean(x)),
            "median_r2": float(np.median(x)),
            "p95_r2": float(np.quantile(x, 0.95)),
            "frac_r2_ge_thr": float(np.mean(x >= r2_threshold)),
        }

    s_before = summarize(r2_before)
    s_after = summarize(r2_after)

    out_tsv = out_prefix + "_ld_validation.tsv"
    pd.DataFrame([
        {"set": "before_prune", **s_before},
        {"set": "after_prune",  **s_after},
    ]).to_csv(out_tsv, sep="\t", index=False)
    print(f"Saved LD validation summary → {out_tsv}")

    # histogram plot
    plt.figure(figsize=(6, 4))
    if r2_before.size > 0:
        plt.hist(r2_before[np.isfinite(r2_before)], bins=40, alpha=0.6, label="before")
    if r2_after.size > 0:
        plt.hist(r2_after[np.isfinite(r2_after)], bins=40, alpha=0.6, label="after")
    plt.axvline(r2_threshold, linestyle="--")
    plt.xlabel("r² (sampled pairs within window)")
    plt.ylabel("Count")
    plt.title(f"LD pruning validation (window ≤ {max_dist_bp:,} bp)")
    plt.legend()
    plt.tight_layout()
    out_png = out_prefix + "_ld_validation_hist.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved LD validation histogram → {out_png}")


# -------------------------------
# PCA on LD-pruned SNPs
# -------------------------------
def compute_pcs(gt_for_pca, n_pcs=3):
    print(f"Computing {n_pcs} genotype PCs (on LD-pruned SNPs)...")
    g_alt = gt_for_pca.to_n_alt().astype(float)  # n_snps x n_samples

    missing = (g_alt < 0)
    g_alt[missing] = np.nan
    snp_means = np.nanmean(g_alt, axis=1)
    g_alt = np.where(np.isnan(g_alt), snp_means[:, None], g_alt)

    pca_result = allel.pca(g_alt, n_components=n_pcs, scaler="patterson")

    var_ratio = None
    if isinstance(pca_result, tuple) and len(pca_result) == 2:
        coords, model = pca_result
        if hasattr(model, "explained_variance_ratio_"):
            var_ratio = np.array(model.explained_variance_ratio_)
    elif isinstance(pca_result, tuple) and len(pca_result) == 3:
        coords, vals, _ = pca_result
        vals = np.array(vals)
        if np.all(np.isfinite(vals)) and vals.sum() > 0:
            var_ratio = vals / vals.sum()
    else:
        raise ValueError(f"Unexpected allel.pca return length: {len(pca_result)}")

    pc_df = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(n_pcs)])
    return pc_df, var_ratio


def plot_pcs(pcs, merged, out_png, var_ratio=None, color_col=META_ECO_COL):
    df = pd.concat([pcs.reset_index(drop=True),
                    merged[[color_col]].reset_index(drop=True)], axis=1)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=df,
        x="PC1",
        y="PC2",
        hue=color_col,
        s=50,
        edgecolor="black",
        linewidth=0.2
    )

    subtitle = ""
    if var_ratio is not None and len(var_ratio) >= 2 and np.all(np.isfinite(var_ratio[:2])):
        subtitle = (
            f"Variance explained: PC1 {var_ratio[0]*100:.1f}%, "
            f"PC2 {var_ratio[1]*100:.1f}% (cum {(var_ratio[0]+var_ratio[1])*100:.1f}%)"
        )

    title = "Genotype PCA (LD-pruned): PC1 vs PC2"
    if subtitle:
        title += "\n" + subtitle

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved PC1 vs PC2 plot → {out_png}")


# -------------------------------
# Phenotype QC
# -------------------------------
def summarize_phenotypes(pheno_eco):
    rows = []
    for trait in TRAITS:
        vals = pheno_eco[trait].values.astype(float)
        rows.append({
            "trait": trait,
            "n_ecotypes": vals.size,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "phenotype_summary.tsv")
    df.to_csv(out, sep="\t", index=False)
    print(f"Saved phenotype summary → {out}")

    for trait in TRAITS:
        plt.figure(figsize=(4, 3))
        sns.histplot(pheno_eco[trait], kde=False)
        plt.xlabel(trait)
        plt.ylabel("Count (ecotypes)")
        plt.title(f"{trait} (ecotype means)")
        plt.tight_layout()
        png = os.path.join(OUT_DIR, f"pheno_hist_{trait}.png")
        plt.savefig(png, dpi=300)
        plt.close()

    corr = pheno_eco[TRAITS].corr()
    corr_tsv = os.path.join(OUT_DIR, "phenotype_trait_correlation.tsv")
    corr.to_csv(corr_tsv, sep="\t")
    print(f"Saved phenotype correlation matrix → {corr_tsv}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", square=True)
    plt.title("Phenotype correlations (ecotype means)")
    plt.tight_layout()
    png = os.path.join(OUT_DIR, "phenotype_trait_correlation.png")
    plt.savefig(png, dpi=300)
    plt.close()


def plot_trait_by_ecotype(pheno_eco):
    for trait in TRAITS:
        plt.figure(figsize=(6, 4))
        sns.barplot(
            data=pheno_eco,
            x=PHENO_ECO_COL,
            y=trait,
            edgecolor="black"
        )
        plt.xlabel("Ecotype")
        plt.ylabel(trait)
        plt.title(f"{trait} by ecotype (mean)")
        plt.tight_layout()
        png = os.path.join(OUT_DIR, f"pheno_{trait}_by_ecotype.png")
        plt.savefig(png, dpi=300)
        plt.close()


# -------------------------------
# Trait–PC diagnostics
# -------------------------------
def trait_pc_regression(merged, pcs, n_pcs=N_PCS):
    rows = []
    X_base = pcs.iloc[:, :n_pcs].values
    X_base = np.column_stack([np.ones(X_base.shape[0]), X_base])  # intercept

    for trait in TRAITS:
        if trait not in merged.columns:
            continue
        y_all = merged[trait].values.astype(float)
        mask = np.isfinite(y_all)
        y = y_all[mask]
        X = X_base[mask, :]

        if y.size < n_pcs + 2:
            continue

        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ss_tot = np.sum((y - y.mean())**2)
        ss_res = np.sum((y - y_hat)**2)
        r2 = 1.0 - ss_res / ss_tot

        row = {
            "trait": trait,
            "n_samples": int(y.size),
            "R2_trait_on_PCs": float(r2),
            "beta_intercept": float(beta[0]),
        }
        for i in range(n_pcs):
            row[f"beta_PC{i+1}"] = float(beta[i+1])
        rows.append(row)

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "trait_pc_regression.tsv")
    df.to_csv(out, sep="\t", index=False)
    print(f"Saved trait–PC regression summary → {out}")


def plot_trait_vs_pcs(merged, pcs):
    df = pd.concat([pcs.reset_index(drop=True),
                    merged[[META_ECO_COL] + TRAITS].reset_index(drop=True)],
                   axis=1)
    for trait in TRAITS:
        if trait not in df.columns:
            continue

        for pc in ["PC1", "PC2"]:
            plt.figure(figsize=(5, 4))
            sns.scatterplot(
                data=df,
                x=pc,
                y=trait,
                hue=META_ECO_COL,
                s=50,
                edgecolor="black",
                linewidth=0.2
            )
            plt.xlabel(pc)
            plt.ylabel(trait)
            plt.title(f"{trait} vs {pc}")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="ecotype")
            plt.tight_layout()
            png = os.path.join(OUT_DIR, f"{trait}_vs_{pc}.png")
            plt.savefig(png, dpi=300)
            plt.close()


# -------------------------------
# Manhattan plot
# -------------------------------
def plot_manhattan(res_df, out_png, title="Manhattan plot",
                   max_labels=20, min_snps_for_label=1000):
    df = res_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["pval"])
    df["neglogp"] = -np.log10(df["pval"])

    chroms = df["CHROM"].astype(str).unique()
    current_offset = 0

    contig_mids = []
    contig_labels = []
    contig_nsnp = []

    df["cum_pos"] = np.nan

    for chrom in chroms:
        idx = df["CHROM"].astype(str) == chrom
        sub = df[idx].sort_values("POS")
        if sub.empty:
            continue

        df.loc[sub.index, "cum_pos"] = sub["POS"] + current_offset

        mid = sub["POS"].median() + current_offset
        contig_mids.append(mid)
        contig_labels.append(chrom)
        contig_nsnp.append(len(sub))

        current_offset += sub["POS"].max()

    contig_mids = np.array(contig_mids)
    contig_labels = np.array(contig_labels)
    contig_nsnp = np.array(contig_nsnp)

    candidate_idx = np.where(contig_nsnp >= min_snps_for_label)[0]

    if candidate_idx.size == 0:
        step = max(1, len(contig_labels) // max_labels)
        label_idx = np.arange(0, len(contig_labels), step)
    else:
        if candidate_idx.size > max_labels:
            order = np.argsort(contig_nsnp[candidate_idx])[::-1]
            label_idx = candidate_idx[order[:max_labels]]
        else:
            label_idx = candidate_idx

    x_ticks_plot = contig_mids[label_idx]
    x_labels_plot = contig_labels[label_idx]

    plt.figure(figsize=(12, 4))
    sns.scatterplot(
        x="cum_pos",
        y="neglogp",
        data=df,
        s=4,
        hue="CHROM",
        palette="tab20",
        legend=False,
    )
    plt.xlabel("Genomic position")
    plt.ylabel("-log10 p-value")
    plt.title(title)
    plt.xticks(x_ticks_plot, x_labels_plot, rotation=90)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# -------------------------------
# Regions of interest
# -------------------------------
def define_regions_of_interest(res_df, window_bp=50_000):
    sig = res_df[res_df["significant_fdr_0_05"]].copy()
    if sig.empty:
        return pd.DataFrame(columns=["CHROM", "start", "end", "n_snps"])

    sig = sig.sort_values(["CHROM", "POS"])

    regions = []
    current = None

    for _, row in sig.iterrows():
        chrom, pos = row["CHROM"], row["POS"]
        if current is None:
            current = {"CHROM": chrom, "start": pos, "end": pos, "n_snps": 1}
        else:
            if chrom == current["CHROM"] and pos - current["end"] <= window_bp:
                current["end"] = pos
                current["n_snps"] += 1
            else:
                regions.append(current)
                current = {"CHROM": chrom, "start": pos, "end": pos, "n_snps": 1}

    if current is not None:
        regions.append(current)

    return pd.DataFrame(regions)


# -------------------------------
# Leave-one-ecotype-out for top SNPs
# -------------------------------
def loo_top_snps(trait_name, res_df, g_alt_trait, merged_with_pcs,
                 sample_mask, n_pcs, top_n=TOP_SNPS_LOO):
    valid_idx = np.where(np.isfinite(res_df["pval"].values))[0]
    if valid_idx.size == 0:
        return

    order = np.argsort(res_df["pval"].values[valid_idx])
    top_idx = valid_idx[order[:min(top_n, order.size)]]

    eco_all = merged_with_pcs[META_ECO_COL].values
    eco_used = eco_all[sample_mask]
    unique_ecos = np.unique(eco_used)

    rows = []

    for s in top_idx:
        chrom = res_df.loc[s, "CHROM"]
        pos = res_df.loc[s, "POS"]
        pval_orig = res_df.loc[s, "pval"]
        beta_orig = res_df.loc[s, "beta"]

        for eco in unique_ecos:
            keep = (eco_used != eco)
            y = merged_with_pcs.loc[sample_mask, trait_name].values.astype(float)[keep]

            if y.size < (n_pcs + 3):
                continue

            C_sub = np.column_stack([
                np.ones(y.shape[0]),
                merged_with_pcs.loc[sample_mask, [f"PC{i+1}" for i in range(n_pcs)]].values[keep, :]
            ])

            g = g_alt_trait[s, keep].astype(float)
            nonmiss = (g >= 0)
            if np.sum(nonmiss) < 5:
                continue
            m = g[nonmiss].mean()
            g[~nonmiss] = m

            CTC = C_sub.T @ C_sub
            CTC_inv = np.linalg.inv(CTC)

            Hy = C_sub @ (CTC_inv @ (C_sub.T @ y))
            r_y = y - Hy
            Syy = np.dot(r_y, r_y)
            df = y.size - C_sub.shape[1] - 1
            if df <= 0:
                continue

            Cg = C_sub.T @ g
            alpha_g = CTC_inv @ Cg
            Hg = C_sub @ alpha_g
            r_g = g - Hg

            Sxx = np.dot(r_g, r_g)
            if Sxx <= 1e-12:
                continue
            Sxy = np.dot(r_g, r_y)
            beta = Sxy / Sxx

            SSE = Syy - (Sxy * Sxy / Sxx)
            if SSE <= 0:
                pval = 1e-12
            else:
                s2 = SSE / df
                se_beta = np.sqrt(s2 / Sxx)
                tstat = beta / se_beta
                pval = 2 * t_dist.sf(np.abs(tstat), df)

            rows.append({
                "trait": trait_name,
                "CHROM": chrom,
                "POS": pos,
                "beta_original": beta_orig,
                "pval_original": pval_orig,
                "ecotype_left_out": eco,
                "n_samples_used": int(y.size),
                "beta_LOO": float(beta),
                "pval_LOO": float(pval),
            })

    if rows:
        df = pd.DataFrame(rows)
        out = os.path.join(OUT_DIR, f"gwas_{trait_name}_top{top_n}_LOO.tsv")
        df.to_csv(out, sep="\t", index=False)
        print(f"  Saved leave-one-ecotype-out table → {out}")


# -------------------------------
# GWAS for a single trait
# -------------------------------
def run_gwas_for_trait(trait_name, merged, gt_filt, chrom, pos, pcs, out_dir,
                       n_pcs=N_PCS, max_threads=1):
    print(f"\n=== GWAS for trait: {trait_name} ===")

    merged_with_pcs = pd.concat([merged.reset_index(drop=True),
                                 pcs.reset_index(drop=True)], axis=1)

    y_all = merged_with_pcs[trait_name].values.astype(float)
    sample_mask = np.isfinite(y_all)
    y = y_all[sample_mask]

    if y.size < 10:
        print(f"  Not enough non-missing values for trait {trait_name} (n={y.size}). Skipping.")
        return None

    C = np.column_stack([
        np.ones(y.shape[0]),
        merged_with_pcs.loc[sample_mask, [f"PC{i+1}" for i in range(n_pcs)]].values
    ])

    n_used = y.shape[0]
    p_C = C.shape[1]

    print(f"  N samples used for {trait_name}: {n_used}")
    print(f"  Covariates: intercept + {n_pcs} PCs (p_C={p_C})")

    CTC = C.T @ C
    CTC_inv = np.linalg.inv(CTC)

    Hy = C @ (CTC_inv @ (C.T @ y))
    r_y = y - Hy
    Syy = np.dot(r_y, r_y)
    df = n_used - p_C - 1
    if df <= 0:
        raise ValueError(f"Non-positive degrees of freedom for trait {trait_name}: df={df}")

    g_alt_all = gt_filt.to_n_alt().astype(float)  # n_snps x n_samples_total
    g_alt_trait = g_alt_all[:, sample_mask]       # n_snps x n_used
    n_snps = g_alt_trait.shape[0]

    print(f"  Running association on {n_snps:,} SNPs (partial regression with PCs)...")

    def assoc_for_snp(s):
        g = g_alt_trait[s, :].copy()
        nonmiss = (g >= 0)
        if np.sum(nonmiss) < 5:
            return np.nan, 1.0, 0.0

        m = g[nonmiss].mean()
        g[~nonmiss] = m

        Cg = C.T @ g
        alpha_g = CTC_inv @ Cg
        Hg = C @ alpha_g
        r_g = g - Hg

        Sxx = np.dot(r_g, r_g)
        if Sxx <= 1e-12:
            return np.nan, 1.0, 0.0

        Sxy = np.dot(r_g, r_y)
        beta = Sxy / Sxx

        SSE = Syy - (Sxy * Sxy / Sxx)
        if SSE <= 0:
            return beta, 1e-12, np.sign(beta) * np.sqrt(Syy / max(Sxx, 1e-12))

        s2 = SSE / df
        se_beta = np.sqrt(s2 / Sxx)
        tstat = beta / se_beta
        pval = 2 * t_dist.sf(np.abs(tstat), df)
        return beta, pval, tstat

    results = Parallel(n_jobs=max_threads)(
        delayed(assoc_for_snp)(s) for s in range(n_snps)
    )

    betas  = np.array([r[0] for r in results])
    pvals  = np.array([r[1] for r in results])
    tstats = np.array([r[2] for r in results])

    rej, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    res_df = pd.DataFrame({
        "CHROM": chrom,
        "POS": pos,
        "beta": betas,
        "tstat": tstats,
        "pval": pvals,
        "pval_fdr": pvals_fdr,
        "significant_fdr_0_05": rej
    })

    out_tsv = os.path.join(out_dir, f"gwas_{trait_name}.tsv")
    res_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"  Saved GWAS results → {out_tsv}")

    manhattan_png = os.path.join(out_dir, f"gwas_{trait_name}_manhattan.png")
    plot_manhattan(res_df, manhattan_png, title=f"GWAS for {trait_name}")
    print(f"  Saved Manhattan plot → {manhattan_png}")

    valid = np.isfinite(pvals) & (pvals > 0) & (pvals <= 1)
    t_valid = tstats[valid]
    chi2_vals = t_valid ** 2
    if chi2_vals.size > 0:
        lambda_gc = float(np.median(chi2_vals) / 0.4549364)  # median chi2_1
    else:
        lambda_gc = np.nan
    print(f"  Genomic inflation factor λGC for {trait_name}: {lambda_gc:.3f}")

    regions_df = define_regions_of_interest(res_df, window_bp=REGION_WINDOW_BP)
    out_regions = os.path.join(out_dir, f"gwas_{trait_name}_regions.tsv")
    regions_df.to_csv(out_regions, sep="\t", index=False)
    print(f"  Saved regions of interest → {out_regions}")

    loo_top_snps(trait_name, res_df, g_alt_trait, merged_with_pcs,
                 sample_mask, n_pcs=n_pcs, top_n=TOP_SNPS_LOO)

    return {
        "trait": trait_name,
        "n_snps_tested": int(n_snps),
        "n_sig_fdr_0_05": int(np.sum(rej)),
        "n_regions": int(regions_df.shape[0]),
        "lambda_gc": lambda_gc,
        "n_samples_used": int(n_used),
    }


# -------------------------------
# MAIN
# -------------------------------
def main():
    meta, vcf_samples, gt, chrom, pos = load_vcf_and_meta()

    merged, vcf_sample_order, pheno_eco = load_and_merge_phenotypes(meta, vcf_samples)

    summarize_phenotypes(pheno_eco)
    plot_trait_by_ecotype(pheno_eco)

    # Reorder genotype array to match merged samples
    order_idx = []
    for sid in merged[META_SAMPLE_COL]:
        idx = np.where(vcf_samples == sid)[0]
        if len(idx) == 0:
            raise ValueError(f"Sample {sid} not found in VCF after alignment.")
        order_idx.append(idx[0])
    order_idx = np.array(order_idx)
    gt = gt[:, order_idx, :]

    # SNP filters (MAF + missingness)
    keep_snps, maf, miss_rate = filter_snps(
        gt,
        maf_threshold=MAF_THRESHOLD,
        miss_threshold=MISS_THRESHOLD
    )
    gt_filt = gt[keep_snps]
    chrom_filt = chrom[keep_snps]
    pos_filt = pos[keep_snps]

    # ---- LD analysis on filtered SNPs (before pruning) ----
    plot_ld_decay(
        gt_filt, chrom_filt, pos_filt,
        out_png=os.path.join(OUT_DIR, "ld_decay_before_prune.png"),
        max_dist_bp=LD_DECAY_MAX_DIST_BP,
        n_pairs=LD_DECAY_N_PAIRS,
        n_bins=LD_DECAY_N_BINS
    )

    plot_ld_heatmap(
        gt_filt, chrom_filt, pos_filt,
        out_png=os.path.join(OUT_DIR, "ld_heatmap_before_prune.png"),
        target_chrom=None,
        n_snps=LD_HEATMAP_N_SNPS
    )

    # ---- LD pruning for PCA ----
    keep_ld = ld_prune_by_chrom(
        gt_filt, chrom_filt,
        size=LD_PRUNE_SIZE,
        step=LD_PRUNE_STEP,
        threshold=LD_PRUNE_R2_THRESHOLD
    )

    # Validate pruning (does LD drop within short windows?)
    validate_ld_pruning(
        gt_filt, chrom_filt, pos_filt,
        keep_ld_mask=keep_ld,
        out_prefix=os.path.join(OUT_DIR, "ld_prune"),
        max_dist_bp=LD_VALIDATE_MAX_DIST_BP,
        n_pairs=LD_VALIDATE_N_PAIRS,
        r2_threshold=LD_PRUNE_R2_THRESHOLD
    )

    # ---- PCA on LD-pruned SNPs ----
    gt_pca = gt_filt[keep_ld]
    pcs, var_ratio = compute_pcs(gt_pca, n_pcs=N_PCS)

    pcs_png = os.path.join(OUT_DIR, "gwas_pcs_PC1_PC2.png")
    plot_pcs(pcs, merged, pcs_png, var_ratio=var_ratio, color_col=META_ECO_COL)

    # Diagnostics
    trait_pc_regression(merged, pcs, n_pcs=N_PCS)
    plot_trait_vs_pcs(merged, pcs)

    # ---- GWAS on the full filtered SNP set (NOT LD-pruned) ----
    summary_rows = []
    for trait in TRAITS:
        if trait not in merged.columns:
            print(f"\nTrait {trait} not found in merged data. Skipping.")
            continue

        summary = run_gwas_for_trait(
            trait_name=trait,
            merged=merged,
            gt_filt=gt_filt,
            chrom=chrom_filt,
            pos=pos_filt,
            pcs=pcs,
            out_dir=OUT_DIR,
            n_pcs=N_PCS,
            max_threads=MAX_THREADS
        )
        if summary is not None:
            summary_rows.append(summary)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        out = os.path.join(OUT_DIR, "gwas_trait_summary.tsv")
        summary_df.to_csv(out, sep="\t", index=False)
        print(f"\nSaved per-trait GWAS summary → {out}")

    print("\nAll GWAS runs finished.")


if __name__ == "__main__":
    main()
