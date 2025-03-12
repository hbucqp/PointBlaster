import sys

# check python version
if sys.version_info < (3, 9):
    sys.exit("Python 3.9 or higher is required to run this script")

import os
import math
import argparse
import subprocess
import pandas as pd
import numpy as np
from Bio import SeqIO
from cvmblaster.blaster import Blaster
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import logging


# from Bio.Blast import NCBIWWW
# from Bio.Blast.Applications import NcbiblastnCommandline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for mutation types
MUTATION_TYPE_INS = 'ins'
MUTATION_TYPE_DEL = 'del'
MUTATION_TYPE_SUB = 'sub'


@dataclass
class MutationInfo:
    type: str
    name: str
    start: int
    end: int
    seq: str
    ref: str
    shift: int


@dataclass
class GeneHit:
    gene: str
    start: int
    sbjct_seq: str
    query_seq: str
    coverage: float
    identity: float


DEFAULT_SETTINGS = {
    'min_identity': 90,
    'min_coverage': 60,
    'threads': 8
}


@dataclass
class ProcessingConfig:
    """Configuration for genome processing"""
    blastdb: Path
    ref_fasta: Path
    db_mutations: Dict
    genes: List[str]
    rna_genes: List[str]
    min_coverage: float
    min_identity: float
    threads: int
    output_path: Path


SUPPORTED_SPECIES = {'salmonella', 'campylobacter'}


def args_parse():
    "Parse the input argument, use '-h' for help."
    parser = argparse.ArgumentParser(
        usage='PointBlaster -i <genome assemble directory> -s <species for point mutation detection> -o <output_directory> \n\nAuthor: Qingpo Cui(SZQ Lab, China Agricultural University)\n')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-i", help="<input_path>: the PATH to the directory of assembled genome files. Could not use with -f")
    group.add_argument(
        "-f", help="<input_file>: the PATH of assembled genome file. Could not use with -i")
    parser.add_argument("-o", help="<output_directory>: output PATH")
    parser.add_argument('-s', type=str, default='salmonella', choices=['salmonella', 'campylobacter'],
                        help='<species>: optional var is [salmonella, campylobacter], other species will be supported soon')
    parser.add_argument('-minid', default=90,
                        help="<minimum threshold of identity>, default=90")
    parser.add_argument('-mincov', default=60,
                        help="<minimum threshold of coverage>, default=60")
    parser.add_argument('-list', action='store_true',
                        help='<show species list>')
    parser.add_argument(
        '-t', default=8, help='<number of threads>: default=8')
    # parser.add_argument("-store_arg_seq", default=False, action="store_true",
    #                     help='<save the nucleotide and amino acid sequence of find genes on genome>')
    # parser.add_argument("-p", default=True, help="True of False to process something",
    #                     type=lambda x: bool(strtobool(str(x).lower())))
    parser.add_argument('-v', '--version', action='version',
                        version='Version: ' + get_version("__init__.py"), help='<display version>')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-init', action='store_true',
                       help='<initialize the point mutationdatabase>')

    # group.add_argument('-updatedb', help="<add input fasta to BLAST database>")
    # group.add_argument('-init', action='store_true',
    #                    help='<initialize the reference database>')
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return parser.parse_args()


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


# def makeblastdb(file, name):
#     cline = NcbimakeblastdbCommandline(
#         dbtype="nucl", out=name, input_file=file)
#     print(f'Making {name} database...')
#     stdout, stderr = cline()
#     print('Finish')


def get_sbjct_seq(ref_fasta: str, gene: str) -> str:
    """Get reference sequence for a gene from FASTA file."""
    with open(ref_fasta) as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            if record.id.split('___')[0] == gene:
                return str(record.seq)
    raise ValueError(f"Gene {gene} not found in reference")


def merge_sequences(first_hit: Tuple, second_hit: Tuple, raw_sbjct_seq: str) -> Tuple:
    """Merge overlapping sequence hits.

    Args:
        first_hit: Tuple of (start, sbjct_seq, query_seq)
        second_hit: Tuple of (start, sbjct_seq, query_seq)
        raw_sbjct_seq: Original reference sequence
    Returns:
        Tuple of (start_pos, merged_sbjct, merged_query)
    """
    start_pos, first_sbjct, first_query = first_hit
    second_start, second_sbjct, second_query = second_hit

    first_end = start_pos + len(first_sbjct) - 1

    if first_end <= second_start:
        gap = raw_sbjct_seq[first_end:second_start - 1]
        merged_sbjct = first_sbjct + gap + second_sbjct
        merged_query = first_query + gap + second_query
    else:
        overlap = first_end - second_start + 1
        merged_sbjct = first_sbjct + second_sbjct[overlap:]
        merged_query = first_query + second_query[overlap:]

    return start_pos, merged_sbjct, merged_query


def get_align_seq(result_dict, ref_fasta):
    """
    Convert BLAST results to gene alignments.
    Returns dict mapping genes to their alignment info:
    {gene: (gene, start, sbjct_seq, query_seq, coverage, identity)}
    """
    gene_list = {}

    # First pass - collect all hits per gene
    gene_hits = {}
    for item_id, hit in result_dict.items():
        gene = hit['GENE']
        if gene not in gene_hits:
            gene_hits[gene] = []
        gene_hits[gene].append((
            hit['SBJSTART'],
            hit['SBJCT_SEQ'],
            hit['QUERY_SEQ']
        ))

    # Process each gene's hits
    for gene, hits in gene_hits.items():
        if len(hits) == 1:
            # Single hit - store directly
            hit = hits[0]
            gene_list[gene] = (
                gene,
                hit[0],  # start
                hit[1],  # sbjct_seq
                hit[2],  # query_seq
                100,     # Default coverage for single hit
                compute_identity(hit[2], hit[1])
            )
        else:
            # Multiple hits - merge in order of start position
            hits.sort(key=lambda x: x[0])
            raw_sbjct_seq = get_sbjct_seq(ref_fasta, gene)

            # Start with first hit
            current = hits[0]

            # Merge subsequent hits
            for next_hit in hits[1:]:
                current = merge_sequences(current, next_hit, raw_sbjct_seq)

            # Store merged result
            coverage = len(current[1]) * 100 / len(raw_sbjct_seq)
            gene_list[gene] = (
                gene,
                current[0],
                current[1],
                current[2],
                coverage,
                compute_identity(current[2], current[1])
            )

    return gene_list


def compute_identity(query_string: str, sbjct_string: str) -> float:
    """Compute sequence identity percentage."""
    matches = sum(q == s for q, s in zip(query_string, sbjct_string))
    return matches * 100 / len(sbjct_string)


def aa(codon):
    """
    This function converts a codon to an amino acid. If the codon is not
    valid an error message is given, or else, the amino acid is returned.
    """
    codon = codon.upper()
    aa = {"ATT": "I", "ATC": "I", "ATA": "I",
          "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L", "TTA": "L", "TTG": "L",
          "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
          "TTT": "F", "TTC": "F",
          "ATG": "M",
          "TGT": "C", "TGC": "C",
          "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
          "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
          "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
          "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
          "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
          "TAT": "Y", "TAC": "Y",
          "TGG": "W",
          "CAA": "Q", "CAG": "Q",
          "AAT": "N", "AAC": "N",
          "CAT": "H", "CAC": "H",
          "GAA": "E", "GAG": "E",
          "GAT": "D", "GAC": "D",
          "AAA": "K", "AAG": "K",
          "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
          "TAA": "*", "TAG": "*", "TGA": "*"}

    # Translate valid codon
    try:
        amino_a = aa[codon]
    except KeyError:
        amino_a = "?"
    return amino_a


# aa = {"ATT": "I", "ATC": "I", "ATA": "I",
#           "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L", "TTA": "L", "TTG": "L",
#           "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
#           "TTT": "F", "TTC": "F",
#           "ATG": "M",
#           "TGT": "C", "TGC": "C",
#           "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
#           "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
#           "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
#           "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
#           "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
#           "TAT": "Y", "TAC": "Y",
#           "TGG": "W",
#           "CAA": "Q", "CAG": "Q",
#           "AAT": "N", "AAC": "N",
#           "CAT": "H", "CAC": "H",
#           "GAA": "E", "GAG": "E",
#           "GAT": "D", "GAC": "D",
#           "AAA": "K", "AAG": "K",
#           "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
#           "TAA": "*", "TAG": "*", "TGA": "*"}


def get_indel(gapped_seq, indel_seq):
    """
    This function finds the zone of gaps compared to the indel sequece.

    """
    insert_seq = indel_seq[0]
    for item in range(1, len(gapped_seq)):
        if gapped_seq[item] == '-':
            insert_seq += indel_seq[item]
        else:
            break
    return insert_seq


def get_substitution(sbjct_seq, substitution_seq):
    """
    This function find substitution zone from query sequence
    """
    sub_seq = substitution_seq[0]
    for item in range(1, len(sbjct_seq)):
        if sbjct_seq[item] != substitution_seq[item]:
            sub_seq += substitution_seq[item]
        else:
            break
    return sub_seq


def process_indel(sbjct_string: str, query_string: str, seq_pos: int, i: int) -> MutationInfo:
    """Process insertion/deletion mutations."""
    is_insertion = sbjct_string[i] == '-'
    mutation_type = MUTATION_TYPE_INS if is_insertion else MUTATION_TYPE_DEL

    indel = get_indel(
        sbjct_string[i:] if is_insertion else query_string[i:],
        query_string[i:] if is_insertion else sbjct_string[i:]
    )

    start_pos = seq_pos
    end_pos = seq_pos + len(indel) - 1

    name = (f"{start_pos}del{indel}" if len(indel) == 1 and mutation_type == MUTATION_TYPE_DEL
            else f"{start_pos}_{end_pos}{mutation_type}{indel}")

    return MutationInfo(
        type=mutation_type,
        name=name,
        start=start_pos,
        end=end_pos,
        seq=indel,
        ref=indel,
        shift=len(indel) - 1
    )


def get_codon_seqs(sbjct_string, query_string, i, seq_pos, sub_seq, sub_end_pos):
    """Get proper codon sequences based on position in codon"""
    pos_in_codon = seq_pos % 3
    padding = 0 if sub_end_pos % 3 == 0 else 3 - (sub_end_pos % 3)

    if pos_in_codon == 0:  # First position
        start_offset = -2
    elif pos_in_codon == 1:  # Second position
        start_offset = 0
    else:  # Third position
        start_offset = -1

    # Raw end_pos is the end position of the substitution sequence
    # end_pos = i + len(sub_seq) - 1 + padding + 1
    # Add 1 and then minus 1 so we could change the code to following:
    end_pos = i + len(sub_seq) + padding

    # Get the codon sequences from reference and query sequences
    ref_seq = sbjct_string[i + start_offset:end_pos]
    query_seq = query_string[i + start_offset:end_pos]

    return ref_seq, query_seq


def process_substitution(sbjct_string: str, query_string: str, i: int, seq_pos: int, gene: str, genes_list: List[str]) -> MutationInfo:
    """Process substitution mutations"""
    sub_seq = get_substitution(sbjct_string[i:], query_string[i:])
    sub_start_pos = seq_pos
    sub_end_pos = seq_pos + len(sub_seq) - 1

    if gene in genes_list:
        ref_seq, query_seq = get_codon_seqs(
            sbjct_string, query_string, i, seq_pos, sub_seq, sub_end_pos)
    else:
        # If the gene is not in the genes_list, use the single nucleotide sequences
        ref_seq = sbjct_string[i:i + len(sub_seq)]
        query_seq = query_string[i:i + len(sub_seq)]

    mutation_name = (f"{seq_pos}{sbjct_string[i]}->{query_string[i]}"
                    if len(sub_seq) == 1
                    else f"{seq_pos}_{sub_end_pos}{ref_seq}->{query_seq}")

    return MutationInfo(
        type=MUTATION_TYPE_SUB,
        name=mutation_name,
        start=sub_start_pos,
        end=sub_end_pos,
        seq=query_seq,
        ref=ref_seq,
        shift=len(sub_seq) - 1
    )


def find_mismatch(sbjct_start, sbjct_string, query_string, gene, genes_list):
    """Find mutations between query and subject sequences"""
    mutations = []
    shift = 0
    seq_pos = sbjct_start

    for index in range(sbjct_start - 1, len(sbjct_string)):
        i = index + shift

        if i >= len(sbjct_string):
            break

        sbjct_nuc = sbjct_string[i].upper()
        query_nuc = query_string[i].upper()

        if sbjct_nuc == query_nuc:
            seq_pos += 1
            continue

        # Process mutation
        if sbjct_nuc == '-' or query_nuc == '-':
            mutation = process_indel(sbjct_string, query_string, seq_pos, i)
        else:
            mutation = process_substitution(
                sbjct_string, query_string, i, seq_pos, gene, genes_list)

        mutations.append([
            mutation.type,  # Access dataclass attributes directly
            mutation.name,
            mutation.start,
            mutation.end,
            mutation.seq,
            mutation.ref
        ])

        # Calculate shift
        shift += mutation.shift  # Access shift directly

        # Calculate new seq_pos
        if mutation.type == MUTATION_TYPE_INS:
            seq_pos = seq_pos
        elif mutation.type == MUTATION_TYPE_DEL:
            seq_pos = seq_pos + mutation.shift
        else:
            seq_pos += 1

    return mutations


def get_db_mutations(mut_db_path):
    """
    transform the table of resistance_overview.txt to dict format
    """
    try:
        drugfile = open(mut_db_path, 'r')
    except:
        sys.exit('Could not found database: %s' % (mut_db_path))

    # Initiate a empty dict
    mutation_dict = {}
    # Go throug mutation file line by line
    for line in drugfile:
        # Ignore headers and check where the indel section starts
        if line.startswith("#"):
            # print(line)
            if "indel" in line.lower():
                indelflag = True
            elif "stop codon" in line.lower():
                stopcodonflag = True
            else:
                stopcodonflag = False
            continue
        # Ignore empty lines
        elif line.strip() == "":
            continue
        else:

            # Strip data entries
            mutation = [data.strip() for data in line.strip().split("\t")]
            # print(mutation)

            # Extract all info on the line (even though it is not all used)
            gene_ID = mutation[0]
            if gene_ID not in mutation_dict.keys():
                mutation_dict[gene_ID] = [{'gene_name': mutation[1], 'mut_pos': int(mutation[2]), 'ref_codon': mutation[3], 'ref_aa': mutation[4], 'alt_aa': mutation[5].split(
                    ","), 'res_drug': mutation[6].replace("\t", " "), 'pmid': mutation[7].split(",")}]
            else:
                mutation_dict[gene_ID] += [{'gene_name': mutation[1], 'mut_pos': int(mutation[2]), 'ref_codon': mutation[3], 'ref_aa': mutation[4], 'alt_aa': mutation[5].split(
                    ","), 'res_drug': mutation[6].replace("\t", " "), 'pmid': mutation[7].split(",")}]

    return mutation_dict


# print(
#
#
#
#
#
#
#
# (1,  'AAATCAGATATAC', 'AAATCAGGATAAC'))
# print(find_mismatch(1,  'AT-GGATC', 'ATCGGATC'))
# print(find_mismatch(1,  'ATCGGATC', 'AT-GGATC'))
# print(find_mismatch(1,  'ATCGAATC', 'ATCAAATC'))


def get_gene_list(species):
    """
    This function return gene list from point_mutation database using species parameter.

    """
    genes_file = os.path.join(
        os.path.dirname(__file__), f'db/point_mutation/{species}/genes.txt')
    RNA_genes_file = os.path.join(
        os.path.dirname(__file__), f'db/point_mutation/{species}/RNA_genes.txt')
    genes = []
    RNA_genes = []
    with open(genes_file, 'r') as f1:
        for i in f1.readlines():
            if i != '':
                genes.append(i.strip())
    with open(RNA_genes_file, 'r') as f2:
        for i in f2.readlines():
            if i != '':
                RNA_genes.append(i.strip())
    for item in RNA_genes:
        if item in genes:
            genes.remove(item)
    return genes, RNA_genes


# print(get_gene_list('salmonella'))


def find_mutations(gene_list_result, genes_list):
    """
    find mutations from gene_list_result
    gene_list_result = [(gene, sbjct_start, sbjct_string,
                           query_string, coverage, identity)]

    """
    mutation_result = {}
    # print(gene_list_result)
    # print(len(gene_list_result))
    for item in gene_list_result:
        coverage = float(item[4])
        # print(f'coverage is {coverage}')
        identity = float(item[5])
        if (coverage == 100.00) or (identity != 100):
            mutation_result[item[0]] = find_mismatch(
                item[1], item[2], item[3], item[0], genes_list)
            )

    return mutation_result


def get_aa_seq(ref_seq, query_seq):
    """
    switch dna sequence to amino acid sequence.

    string = 'ATCATG'
    for i in np.arange(0, 6, 3):
        print(i)
        print(string[i:i + 3])

    """
    aa_ref=''
    aa_alt=''

    for i in np.arange(0, len(ref_seq), 3):
        aa_ref += aa(ref_seq[i:i + 3])
        aa_alt += aa(query_seq[i:i + 3])
    return aa_ref, aa_alt


def match_mut_indb(db_mutations, gene_name, aa_pos, aa_ref, aa_alt):
    """
    """
    save_check=0
    resistance_phenotype=''
    gene=''
    gene_mut_list=db_mutations[gene_name]

    for single_mut_dict in gene_mut_list:

        if (aa_pos == single_mut_dict['mut_pos']) and (aa_alt in single_mut_dict['alt_aa']) and (aa_ref == single_mut_dict['ref_aa']):
            save_check=1
            gene=single_mut_dict['gene_name']
            resistance_phenotype=single_mut_dict['res_drug']
            # print(save_check, gene, aa_pos, resistance_phenotype)
        else:
            next

    return save_check, gene, resistance_phenotype


def get_rna_change(ref_seq, query_seq, sub_position):
    """
    find substitution nuc in the zone of substitution of rna gene sequence

    """
    sub_nuc_index=sub_position % 3 - 1
    ref_nuc=ref_seq[sub_nuc_index]
    query_nuc=query_seq[sub_nuc_index]
    return ref_nuc, query_nuc


def filter_result(mutation_dict, db_mutations, pm_db_list):
    result=''
    for key in mutation_dict.keys():
        gene_name=key
        # print(key)
        if key in pm_db_list:
            for item in mutation_dict[key]:
                # print(key)
                if item[0] == 'sub':
                    sub_start_pos=item[2]
                    aa_pos=math.ceil(sub_start_pos / 3)
                    aa_ref, aa_alt=get_aa_seq(item[5], item[4])
                    # print(item[2], item[5], aa_ref, aa_alt)
                    # print(item)
                    save, gene, res_pheno=match_mut_indb(
                        db_mutations, gene_name, aa_pos, aa_ref, aa_alt)
                else:
                    save=0
                    # print(save, gene, res_pheno)
                if save:
                        # print(gene_name)
                    result += f'{gene}\t{aa_ref}{aa_pos}{aa_alt}\t{item[5]} -> {item[4]}\t{aa_ref} -> {aa_alt}\t{res_pheno}\n'
                    # print('begin')
                    # print(result)
                    # print('end')

                    # print(aa_pos)
                    # print(aa_ref, aa_alt)
        else:
            for item in mutation_dict[key]:
                if item[0] == 'sub':
                    sub_start_pos=item[2]
                    ref_seq=item[5]
                    alt_seq=item[4]
                    # nuc_ref, nuc_alt = get_rna_change(
                    #     ref_seq, alt_seq, sub_start_pos)
                    # xxx
                    aa_ref=ref_seq
                    aa_alt=alt_seq
                    aa_pos=sub_start_pos
                    save, gene, res_pheno=match_mut_indb(
                        db_mutations, gene_name, aa_pos, aa_ref, aa_alt)
                    # print('RNA')
                    # print(save, gene, res_pheno)
                else:
                    save=0
                if save:
                    # print(gene_name)
                    result += f'{gene}\t{aa_ref}{aa_pos}{aa_alt}\t{item[5]} -> {item[4]}\t{aa_ref} -> {aa_alt}\t{res_pheno}\n'
                    # print('begin')
                    # print(result)
                    # print(end)

    return result


def show_db_list():
    print('Datbase of point mutation')
    db_path=os.path.join(os.path.dirname(__file__), 'db/point_mutation')
    for file in os.listdir(db_path):
        if os.path.isdir(os.path.join(db_path, file)):
            print(file)


def initialize_db():
    """Initialize BLAST databases for point mutation detection using pathlib."""
    database_path=Path(__file__).parent / 'db/point_mutation'

    for point_db_path in database_path.iterdir():
        if not point_db_path.is_dir():
            continue

        point_db_name=point_db_path.name
        fsa_file=point_db_path / f"{point_db_name}.fsa"

        if fsa_file.exists():
            out_path=point_db_path / point_db_name
            print(f'Making {point_db_name} point mutation database...')
            Blaster.makeblastdb(
                file=str(fsa_file),
                name=str(out_path),
                db_type='nucl'
            )


def process_genome_file(file_path: Path, config: ProcessingConfig) -> pd.DataFrame:
    """Process a single genome file and return results DataFrame.

    Args:
        file_path: Path to genome file
        config: Processing configuration
    Returns:
        DataFrame with mutation results
    """
    logger.info(f"Processing {file_path}")

    # Validate input file
    if not file_path.is_file():
        raise FileNotFoundError(f"Genome file not found: {file_path}")
    if not Blaster.is_fasta(file_path):
        raise ValueError(f"Not a valid FASTA file: {file_path}")

    # Run BLAST with lower coverage for 23S mutations
    blaster = Blaster(
        inputfile = file_path,
        database = config.blastdb,
        output = config.output_path,
        threads = config.threads,
        minid = config.min_identity,
        mincov = 20,  # Lower coverage threshold for 23S
        blast_type = 'blastn'
    )
    _, result_dict = blaster.biopython_blast()

    # Process alignments and filter by coverage
    gene_hits = get_align_seq(result_dict, config.ref_fasta)
    gene_hits = {
        gene: hit for gene, hit in gene_hits.items()
        if float(hit[4]) >= config.min_coverage
    }

    # Find mutations
    mutation_result = find_mutations(list(gene_hits.values()), config.genes)

    # Format results
    return format_mutation_results(
        mutation_result,
        config.db_mutations,
        config.genes,
        file_path.stem,
        config.output_path
    )

def format_mutation_results(
    mutation_result: Dict,
    db_mutations: Dict,
    genes: List[str],
    file_name: str,
    output_path: Path
) -> pd.DataFrame:
    """Format mutation results into a DataFrame and write to file.

    Args:
        mutation_result: Dictionary of mutations by gene
        db_mutations: Database of known mutations
        genes: List of protein-coding genes
        file_name: Name of the input file
        output_path: Path to output directory

    Returns:
        DataFrame containing formatted mutation results
    """
    results = []

    for gene_name, mutations in mutation_result.items():
        # Handle protein-coding genes
        if gene_name in genes:
            for mut in mutations:
                if mut[0] == MUTATION_TYPE_SUB:
                    sub_start_pos = mut[2]
                    aa_pos = math.ceil(sub_start_pos / 3)
                    aa_ref, aa_alt = get_aa_seq(mut[5], mut[4])

                    save, gene, res_pheno = match_mut_indb(
                        db_mutations, gene_name, aa_pos, aa_ref, aa_alt
                    )

                    if save:
                        results.append({
                            'File': file_name,
                            'Gene': gene,
                            'Mutation': f"{aa_ref}{aa_pos}{aa_alt}",
                            'Nucleotide_change': f"{mut[5]} -> {mut[4]}",
                            'AA_change': f"{aa_ref} -> {aa_alt}",
                            'Resistance': res_pheno
                        })

        # Handle RNA genes
        else:
            for mut in mutations:
                if mut[0] == MUTATION_TYPE_SUB:
                    sub_start_pos = mut[2]
                    ref_seq = mut[5]
                    alt_seq = mut[4]

                    # For RNA genes, use the nucleotide sequence directly
                    aa_ref = ref_seq
                    aa_alt = alt_seq
                    aa_pos = sub_start_pos

                    save, gene, res_pheno = match_mut_indb(
                        db_mutations, gene_name, aa_pos, aa_ref, aa_alt
                    )

                    if save:
                        results.append({
                            'File': file_name,
                            'Gene': gene,
                            'Mutation': f"{aa_ref}{aa_pos}{aa_alt}",
                            'Nucleotide_change': f"{mut[5]} -> {mut[4]}",
                            'AA_change': f"{aa_ref} -> {aa_alt}",
                            'Resistance': res_pheno
                        })

    # Create DataFrame with default columns even if empty
    df = pd.DataFrame(results, columns=[
        'File', 'Gene', 'Mutation', 'Nucleotide_change',
        'AA_change', 'Resistance'
    ])

    # Write results to file, including headers even if empty
    output_file = output_path / f'{file_name}_out.txt'
    df.to_csv(output_file, sep='\t', index=False)
    if df.empty:
        logger.info(f"No mutations found. Empty results file with headers written to {output_file}")
    else:
        logger.info(f"Results written to {output_file}")

    return df

def process_genome_files(
    input_files: List[Path],
    config: ProcessingConfig
) -> pd.DataFrame:
    """Process multiple genome files and combine results."""
    all_results = []

    for file_path in input_files:
        try:
            results = process_genome_file(file_path, config)
            if not results.empty:
                all_results.append(results)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue

    if not all_results:
        return pd.DataFrame()

    # Combine all results
    df_final = pd.concat(all_results, ignore_index=True)

    # Create pivot table
    df_pivot = df_final.pivot_table(
        index='File',
        columns='Gene',
        values='Mutation',
        aggfunc=lambda x: ','.join(map(str, x))
    )

    return df_pivot

def main():
    args = args_parse()

    if args.list:
        show_db_list()
        sys.exit(0)
    elif args.init:
        initialize_db()
        sys.exit(0)

    # Validate species
    if args.s not in SUPPORTED_SPECIES:
        logger.error(f"Unsupported species. Supported species are: {', '.join(SUPPORTED_SPECIES)}")
        sys.exit(1)

    # Setup paths
    input_path = Path(args.i if args.i else args.f).resolve()
    output_path = Path(args.o).resolve()
    output_path.mkdir(exist_ok=True)

    # Get input files
    input_files = []
    if args.i:
        input_files = list(input_path.glob('*.fa*'))  # Match .fa, .fasta, etc.
    else:
        input_files = [input_path]

    if not input_files:
        logger.error(f"No FASTA files found in {input_path}")
        sys.exit(1)

    # Setup processing configuration
    config = ProcessingConfig(
        blastdb=Path(__file__).parent / f'db/point_mutation/{args.s}/{args.s}',
        ref_fasta=Path(__file__).parent / f'db/point_mutation/{args.s}/{args.s}.fsa',
        db_mutations=get_db_mutations(Path(__file__).parent / f'db/point_mutation/{args.s}/resistens-overview.txt'),
        genes=get_gene_list(args.s)[0],
        rna_genes=get_gene_list(args.s)[1],
        min_coverage=float(args.mincov),
        min_identity=float(args.minid),
        threads=int(args.t),
        output_path=output_path
    )

    # Process files and generate summary
    df_pivot = process_genome_files(input_files, config)

    if not df_pivot.empty:
        summary_file = output_path / 'PointMutation_Summary.csv'
        df_pivot.to_csv(summary_file)
        logger.info(f"Results written to {summary_file}")
    else:
        logger.warning("No mutations found in any input files")

if __name__ == '__main__':
    main()
