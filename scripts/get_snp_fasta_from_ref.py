#!/usr/bin/env python3
"""
Extract reference sequences around GWAS regions.

For each file gwas_<trait>_regions.tsv in --regions_dir, this script:
  - Reads CHROM, start, end, n_snps
  - Expands each region by --window_bp upstream and downstream
  - Extracts the DNA sequence from the reference FASTA
  - Writes one FASTA per trait: gwas_<trait>_regions_±window.fa
"""


# -------------------------------
# CONFIG DEFAULTS
# -------------------------------
DEFAULT_REF_FASTA   = "reference/eriophorum_vaginatum.fa"
DEFAULT_REGIONS_DIR = "data/genomic_data/gwas_results"
DEFAULT_OUT_DIR     = "data/genomic_data/gwas_results/trait_fastas"
DEFAULT_WINDOW_BP   = 2000


# -------------------------------
# Helpers
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Extract reference sequences around GWAS regions for each trait."
    )
    p.add_argument(
        "--ref_fasta",
        default=DEFAULT_REF_FASTA,
        help=f"Reference genome FASTA (default: {DEFAULT_REF_FASTA})",
    )
    p.add_argument(
        "--regions_dir",
        default=DEFAULT_REGIONS_DIR,
        help=f"Directory with gwas_<trait>_regions.tsv files "
             f"(default: {DEFAULT_REGIONS_DIR})",
    )
    p.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for FASTA files (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--window_bp",
        type=int,
        default=DEFAULT_WINDOW_BP,
        help=f"Number of bp upstream and downstream to include (default: {DEFAULT_WINDOW_BP})",
    )
    return p.parse_args()


def load_reference(ref_fasta: str) -> Dict[str, SeqIO.SeqRecord]:
    """
    Load reference FASTA into a dict: {chrom_name: SeqRecord}
    """
    if not os.path.exists(ref_fasta):
        raise FileNotFoundError(f"Reference FASTA not found: {ref_fasta}")

    print(f"Loading reference genome from {ref_fasta} ...")
    ref_dict = SeqIO.to_dict(SeqIO.parse(ref_fasta, "fasta"))
    print(f"  Loaded {len(ref_dict)} contigs from reference.")
    return ref_dict


def infer_trait_from_filename(path: str) -> str:
    """
    Given .../gwas_<trait>_regions.tsv, return <trait>.
    """
    base = os.path.basename(path)
    if not base.startswith("gwas_") or not base.endswith("_regions.tsv"):
        # fall back to stripping extension
        trait = os.path.splitext(base)[0]
        return trait

    trait = base[len("gwas_"):-len("_regions.tsv")]
    return trait


def safe_trait_name(trait: str) -> str:
    """
    Make trait name safe for FASTA headers (no spaces, etc.).
    """
    return trait.replace(" ", "_")


def extract_region_sequence(
    ref_dict: Dict[str, SeqIO.SeqRecord],
    chrom: str,
    start: int,
    end: int,
    window_bp: int
):
    """
    Extract reference sequence for [start - window_bp, end + window_bp], 1-based inclusive
    coordinates, clamped to contig boundaries.

    Returns:
      (seq_str, win_start, win_end)

    Where win_start, win_end are the (1-based inclusive) coordinates actually used.
    """
    if chrom not in ref_dict:
        raise KeyError(f"Chromosome/contig '{chrom}' not found in reference FASTA.")

    seq_record = ref_dict[chrom]
    chrom_len = len(seq_record.seq)

    # Desired window in 1-based coordinates
    win_start = max(1, int(start) - window_bp)
    win_end   = min(chrom_len, int(end) + window_bp)

    # Convert to 0-based Python slice (end-exclusive)
    s0 = win_start - 1
    e0 = win_end

    seq_str = str(seq_record.seq[s0:e0]).upper()
    return seq_str, win_start, win_end


def process_regions_file(
    regions_path: str,
    ref_dict: Dict[str, SeqIO.SeqRecord],
    out_dir: str,
    window_bp: int
):
    """
    Process one gwas_<trait>_regions.tsv and write a FASTA of windows.
    """
    if not os.path.exists(regions_path):
        print(f"Regions file not found (skipping): {regions_path}")
        return

    trait = infer_trait_from_filename(regions_path)
    trait_safe = safe_trait_name(trait)

    print(f"\nProcessing regions for trait '{trait}' from {regions_path}")

    df = pd.read_csv(regions_path, sep="\t")

    # Expect columns: CHROM, start, end, n_snps
    required_cols = {"CHROM", "start", "end"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"  WARNING: Missing required columns {missing} in {regions_path}, skipping.")
        return

    if df.empty:
        print(f"  No regions found for trait {trait} (regions TSV is empty). Skipping FASTA.")
        return

    # Prepare output FASTA
    out_fasta = os.path.join(
        out_dir,
        f"gwas_{trait_safe}_regions_pm{window_bp//1000}kb.fa"
    )
    n_written = 0

    with open(out_fasta, "w") as fasta_out:
        for idx, row in df.iterrows():
            chrom = str(row["CHROM"])
            start = int(row["start"])
            end   = int(row["end"])
            n_snps = int(row["n_snps"]) if "n_snps" in row else None

            try:
                seq_str, win_start, win_end = extract_region_sequence(
                    ref_dict=ref_dict,
                    chrom=chrom,
                    start=start,
                    end=end,
                    window_bp=window_bp
                )
            except KeyError as e:
                print(f"  WARNING: {e} (region row {idx}, {chrom}:{start}-{end}), skipping.")
                continue

            # Build FASTA header
            if n_snps is not None:
                header = (
                    f">trait={trait_safe}|CHROM={chrom}|window={win_start}-{win_end}"
                    f"|region={start}-{end}|n_snps={n_snps}"
                )
            else:
                header = (
                    f">trait={trait_safe}|CHROM={chrom}|window={win_start}-{win_end}"
                    f"|region={start}-{end}"
                )

            fasta_out.write(header + "\n")
            # Wrap sequence to 60 characters per line for readability
            for i in range(0, len(seq_str), 60):
                fasta_out.write(seq_str[i:i+60] + "\n")

            n_written += 1

    print(f"  Wrote {n_written} regions to {out_fasta}")


# -------------------------------
# MAIN
# -------------------------------
def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load reference
    ref_dict = load_reference(args.ref_fasta)

    # Find all trait region TSVs
    region_files = [
        os.path.join(args.regions_dir, f)
        for f in os.listdir(args.regions_dir)
        if f.startswith("gwas_") and f.endswith("_regions.tsv")
    ]

    if not region_files:
        print(f"No gwas_<trait>_regions.tsv files found in {args.regions_dir}")
        return

    print(f"Found {len(region_files)} region files.")

    for regions_path in region_files:
        process_regions_file(
            regions_path=regions_path,
            ref_dict=ref_dict,
            out_dir=args.out_dir,
            window_bp=args.window_bp
        )

    print("\nDone extracting reference windows for all traits.")


if __name__ == "__main__":
    main()
