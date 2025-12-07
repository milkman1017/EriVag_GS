#!/usr/bin/env python3
import os
import csv
import subprocess
from pathlib import Path
from Bio import Entrez

Entrez.email = "milkmanmahler1017@gmail.com"

# Path to your reference genome FASTA (must be bwa-indexed and faidx'd)
REF_FASTA = "reference/eriophorum_vaginatum.fa"

# Threads for mapping / QC / calling
THREADS = "8"

# Base output directory
BASE_OUTDIR = Path("data/genomic_data")

# Whether to run fastp QC
USE_FASTP = True


def run_cmd(cmd, cwd=None, use_shell=False):
    """
    Run a shell command and crash if it fails.
    cmd: list[str] if use_shell=False, or str if use_shell=True.
    """
    if use_shell:
        print("Running (shell):", cmd)
        subprocess.run(cmd, shell=True, check=True, cwd=cwd)
    else:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=cwd)


def get_run_accessions(srx):
    """
    Given an SRX accession, return all SRR run accessions.
    """
    handle = Entrez.esearch(db="sra", term=srx)
    record = Entrez.read(handle)
    handle.close()

    if not record["IdList"]:
        raise ValueError(f"No SRA record found for {srx}")

    sra_id = record["IdList"][0]

    # Try runinfo first
    handle = Entrez.efetch(db="sra", id=sra_id, rettype="runinfo", retmode="text")
    raw = handle.read()
    handle.close()

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")

    lines = raw.strip().splitlines()
    if len(lines) > 1:
        header = lines[0].split(",")
        if "Run" in header:
            idx = header.index("Run")
            runs = [row.split(",")[idx] for row in lines[1:] if row.strip()]
            if runs:
                return runs

    # Fallback: esummary
    handle = Entrez.esummary(db="sra", id=sra_id, retmode="xml")
    summary = Entrez.read(handle)
    handle.close()

    runs = []
    exp = summary[0]
    if "Runs" in exp:
        for r in exp["Runs"].split(","):
            r = r.strip()
            if r.startswith("SRR"):
                runs.append(r)

    if not runs:
        raise RuntimeError(f"No SRR runs found for {srx}")

    return runs


def download_fastq(run_accession, outdir):
    """
    Download one SRR using prefetch + fasterq-dump (single-end).
    Output: <run>.fastq in outdir.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Downloading {run_accession} ===")
    run_cmd(["prefetch", run_accession])
    run_cmd([
        "fasterq-dump",
        "--outdir", str(outdir),
        run_accession
    ])

    fq = outdir / f"{run_accession}.fastq"

    if not fq.exists():
        raise FileNotFoundError(f"Missing {fq}")

    return fq


def qc_fastq(run_accession, fq, outdir):
    """
    Run fastp on single-end reads.
    Output: <run>.clean.fastq
    """
    outdir = Path(outdir)
    clean = outdir / f"{run_accession}.clean.fastq"

    json_report = outdir / f"{run_accession}.fastp.json"
    html_report = outdir / f"{run_accession}.fastp.html"

    print(f"=== QC with fastp: {run_accession} ===")
    run_cmd([
        "fastp",
        "-i", str(fq),
        "-o", str(clean),
        "-w", THREADS,
        "-j", str(json_report),
        "-h", str(html_report)
    ])

    return clean


def align_and_sort(sample_id, fq, outdir, ref_fasta=REF_FASTA):
    """
    Map single-end reads to reference and produce sorted, indexed BAM.

    Output:
      <outdir>/<sample_id>.sorted.bam
      <outdir>/<sample_id>.sorted.bam.bai
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bam_path = outdir / f"{sample_id}.sorted.bam"

    print(f"=== Aligning {sample_id} to reference ===")

    # bwa mem (single-end) -> samtools sort -> BAM
    cmd = (
        f"bwa mem -t {THREADS} {ref_fasta} {fq} "
        f"| samtools sort -@ {THREADS} -o {bam_path} -"
    )
    run_cmd(cmd, use_shell=True)

    run_cmd(["samtools", "index", str(bam_path)])

    return bam_path


def cleanup_fastqs_for_run(run_accession, outdir, used_fastp):
    """
    Delete FASTQ files for one run to save disk space.
    Removes:
      <run>.fastq
      and if fastp used:
      <run>.clean.fastq
    """
    outdir = Path(outdir)
    print(f"=== Cleaning up FASTQs for {run_accession} ===")

    paths_to_delete = [
        outdir / f"{run_accession}.fastq",
    ]

    if used_fastp:
        paths_to_delete.append(outdir / f"{run_accession}.clean.fastq")

    for p in paths_to_delete:
        if p.exists():
            print(f"  deleting {p}")
            p.unlink()
        else:
            print(f"  (not found, skipping) {p}")


def joint_variant_calling(ref_fasta, bam_paths, out_vcf_gz):
    """
    Run bcftools mpileup + bcftools call on all BAMs jointly.
    Output: compressed VCF (.vcf.gz) + index (.csi)
    """
    bam_paths = [str(p) for p in bam_paths]
    out_vcf_gz = Path(out_vcf_gz)

    print("\n=== Joint variant calling with bcftools ===")
    mpileup_cmd = [
        "bcftools", "mpileup", "-Ou",
        "-f", ref_fasta
    ] + bam_paths

    call_cmd = [
        "bcftools", "call", "-mv",
        "-Oz", "-o", str(out_vcf_gz)
    ]

    joined_cmd = (
        " ".join(mpileup_cmd) +
        " | " +
        " ".join(call_cmd)
    )
    run_cmd(joined_cmd, use_shell=True)

    run_cmd(["bcftools", "index", str(out_vcf_gz)])

    print(f"Joint VCF written to: {out_vcf_gz}")


def process_csv(csv_path, max_samples=None):
    """
    Process the input CSV and run the full pipeline.

    If max_samples is not None, only the first N rows of the CSV
    are used (for debugging).
    """
    csv_path = Path(csv_path)

    if not Path(REF_FASTA).exists():
        raise FileNotFoundError(
            f"Reference FASTA {REF_FASTA} not found. "
            f"Set REF_FASTA at the top of this script."
        )

    ecotype_to_srx = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples is not None and i >= max_samples:
                print(f"Reached max_samples={max_samples}, stopping CSV read.")
                break

            srx = row["Experiment Accession"]
            library_name = row["Library Name"]
            ecotype = library_name.split("-")[0]
            ecotype_to_srx.setdefault(ecotype, []).append(srx)

    if max_samples is not None:
        print(f"DEBUG: Using only the first {max_samples} rows from {csv_path}")

    all_bams = []
    metadata_rows = []

    for ecotype, srx_list in ecotype_to_srx.items():
        print("\n===================================================")
        print(f"PROCESSING ECOTYPE: {ecotype}")
        print(f"  SRXs: {srx_list}")
        print("===================================================\n")

        ecotype_dir = BASE_OUTDIR / ecotype
        ecotype_dir.mkdir(parents=True, exist_ok=True)

        for srx in srx_list:
            print(f"\n--- SRX: {srx} ---")
            runs = get_run_accessions(srx)
            print(f"  SRR runs: {runs}")

            for run in runs:
                sample_id = run

                print(f"\n### SAMPLE {sample_id} (ecotype {ecotype}) ###")

                fq = download_fastq(run, ecotype_dir)

                if USE_FASTP:
                    fq_clean = qc_fastq(run, fq, ecotype_dir)
                else:
                    fq_clean = fq

                bam_path = align_and_sort(
                    sample_id=sample_id,
                    fq=fq_clean,
                    outdir=ecotype_dir
                )
                all_bams.append(bam_path)

                cleanup_fastqs_for_run(run, ecotype_dir, used_fastp=USE_FASTP)

                metadata_rows.append({
                    "sample_id": sample_id,
                    "ecotype": ecotype,
                    "SRR": run,
                    "SRX": srx
                })

    BASE_OUTDIR.mkdir(parents=True, exist_ok=True)
    meta_path = BASE_OUTDIR / "sample_metadata.tsv"
    print(f"\nWriting sample metadata to: {meta_path}")
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "ecotype", "SRR", "SRX"],
            delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    if all_bams:
        joint_vcf = BASE_OUTDIR / "all_samples.vcf.gz"
        joint_variant_calling(REF_FASTA, all_bams, joint_vcf)
    else:
        print("No BAMs produced — nothing to call variants on.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Download GBS/RAD data per individual, "
            "align to a single reference, and joint-call SNPs (single-end)."
        )
    )
    parser.add_argument(
        "csv_file",
        help="Input CSV with columns 'Experiment Accession' and 'Library Name'"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="If set, only process the first N rows from the CSV (for debugging)."
    )
    args = parser.parse_args()

    process_csv(args.csv_file, max_samples=args.max_samples)
