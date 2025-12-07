#!/usr/bin/env bash
set -euo pipefail


#Gets eriophorum refernece genome from genbank and indexes it for mapping


# GenBank assembly accession and name
ACCESSION="GCA_965200365.1"
ASM_NAME="lpEriVagi1.hap1.1"

# Output directory for the reference
OUTDIR="reference"

# Final fasta name you’ll point your pipeline at
FINAL_FA="eriophorum_vaginatum.fa"


need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: required command '$1' not found in PATH" >&2
        exit 1
    fi
}

need_cmd samtools
need_cmd bwa

# use either wget or curl
if command -v wget >/dev/null 2>&1; then
    DL_TOOL="wget"
elif command -v curl >/dev/null 2>&1; then
    DL_TOOL="curl"
else
    echo "ERROR: need either 'wget' or 'curl' installed to download the genome." >&2
    exit 1
fi

# NCBI layout for GenBank assemblies:
# ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/XXX/YYY/ZZZ/GCA_XXXYYYZZZ.N_AssemblyName/
# where XXXYYYZZZ are 3-3-3 splits of the numeric part of the accession

NUM="${ACCESSION#GCA_}"   # e.g. 965200365
PART1="${NUM:0:3}"        # 965
PART2="${NUM:3:3}"        # 200
PART3="${NUM:6:3}"        # 365

BASE_URL="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/${PART1}/${PART2}/${PART3}/${ACCESSION}_${ASM_NAME}"
FASTA_GZ="${ACCESSION}_${ASM_NAME}_genomic.fna.gz"


mkdir -p "${OUTDIR}"
cd "${OUTDIR}"

# Download FASTA (if needed)
if [[ -f "${FINAL_FA}" ]]; then
    echo "Found existing ${FINAL_FA} – skipping download."
else
    if [[ -f "${FASTA_GZ}" ]]; then
        echo "Found existing ${FASTA_GZ} – skipping re-download."
    else
        echo "Downloading genome from:"
        echo "  ${BASE_URL}/${FASTA_GZ}"
        if [[ "${DL_TOOL}" == "wget" ]]; then
            wget -O "${FASTA_GZ}" "${BASE_URL}/${FASTA_GZ}"
        else
            curl -L -o "${FASTA_GZ}" "${BASE_URL}/${FASTA_GZ}"
        fi
    fi

    echo "Decompressing ${FASTA_GZ} → ${FINAL_FA}"
    gunzip -c "${FASTA_GZ}" > "${FINAL_FA}"
fi


# Index with bwa and samtools
echo "Running bwa index on ${FINAL_FA}..."
bwa index "${FINAL_FA}"

echo "Running samtools faidx on ${FINAL_FA}..."
samtools faidx "${FINAL_FA}"

echo "Done."
echo "Reference is at: $(pwd)/${FINAL_FA}"
echo "You can now set REF_FASTA=\"reference/${FINAL_FA}\" in your Python pipeline."
