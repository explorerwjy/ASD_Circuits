# Author: jywang explorerwjy@gmail.com
# Enhanced gene matching module for null model generation
# This module provides gene property matching functionality for permutation testing

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MatchingConfig:
    """Configuration for gene property matching."""
    # Matching properties to use
    match_length: bool = True
    match_conservation: bool = True
    match_expression: bool = True

    # Matching stringency (number of bins for each property)
    length_bins: int = 10  # Deciles by default
    conservation_bins: int = 10
    expression_bins: int = 10

    # Tolerance multiplier for bin matching (1.0 = exact bin, 2.0 = ±1 bin)
    tolerance: float = 1.0

    # Minimum number of candidate genes per query gene
    min_candidates: int = 50

    # Whether to use weighted sampling by mutation rate
    weighted_sampling: bool = True


class GenePropertyMatcher:
    """
    Enhanced gene matching based on multiple gene properties.

    This class implements the enhanced null model generation that matches
    on gene length, conservation (phastCons), and expression level to
    control for potential confounds in permutation testing.
    """

    def __init__(
        self,
        gene_annotations: pd.DataFrame,
        expression_matrix: Optional[pd.DataFrame] = None,
        config: Optional[MatchingConfig] = None
    ):
        """
        Initialize the gene property matcher.

        Parameters
        ----------
        gene_annotations : pd.DataFrame
            DataFrame with gene annotations. Must have columns for:
            - gene_id (index or column): Gene identifier (entrez_id or ensembl)
            - cds_length: Coding sequence length
            - phastcons: Conservation score (phastCons)
            Optional:
            - pLI, LOEUF: Constraint scores
            - mutation_rate: Background mutation rate
        expression_matrix : pd.DataFrame, optional
            Gene expression matrix (genes x samples/cell_types)
            Used to compute mean expression levels
        config : MatchingConfig, optional
            Configuration for matching parameters
        """
        self.config = config or MatchingConfig()
        self._gene_annotations = gene_annotations.copy()
        self._expression_matrix = expression_matrix

        # Precompute gene property bins for efficient matching
        self._precompute_bins()

    def _precompute_bins(self):
        """Precompute property bins for all genes."""
        logger.info("Precomputing gene property bins...")

        # Create copy for binning
        df = self._gene_annotations.copy()

        # Bin gene length (log-transformed for better distribution)
        if 'cds_length' in df.columns and self.config.match_length:
            df['log_length'] = np.log10(df['cds_length'].clip(lower=1))
            df['length_bin'] = pd.qcut(
                df['log_length'].rank(method='first'),
                q=self.config.length_bins,
                labels=False,
                duplicates='drop'
            )
        else:
            df['length_bin'] = 0

        # Bin conservation score
        if 'phastcons' in df.columns and self.config.match_conservation:
            df['conservation_bin'] = pd.qcut(
                df['phastcons'].rank(method='first'),
                q=self.config.conservation_bins,
                labels=False,
                duplicates='drop'
            )
        else:
            df['conservation_bin'] = 0

        # Compute and bin mean expression
        if self._expression_matrix is not None and self.config.match_expression:
            # Compute mean expression across all samples
            mean_expr = self._expression_matrix.mean(axis=1)
            # Map to gene annotations (handle different index types)
            df['mean_expression'] = df.index.map(mean_expr)
            df['expression_bin'] = pd.qcut(
                df['mean_expression'].rank(method='first'),
                q=self.config.expression_bins,
                labels=False,
                duplicates='drop'
            ).fillna(-1).astype(int)
        else:
            df['expression_bin'] = 0

        # Create composite bin key for efficient lookup
        df['composite_bin'] = (
            df['length_bin'].astype(str) + '_' +
            df['conservation_bin'].astype(str) + '_' +
            df['expression_bin'].astype(str)
        )

        self._binned_genes = df

        # Create lookup dict: composite_bin -> list of genes
        self._bin_to_genes: Dict[str, List] = {}
        for bin_key, group in df.groupby('composite_bin'):
            self._bin_to_genes[bin_key] = group.index.tolist()

        logger.info(f"Created {len(self._bin_to_genes)} unique bin combinations")

    def get_matched_candidates(
        self,
        query_gene: Union[int, str],
        exclude_genes: Optional[List] = None,
        tolerance: Optional[float] = None
    ) -> List:
        """
        Get candidate genes matching the properties of a query gene.

        Parameters
        ----------
        query_gene : int or str
            Gene identifier to match
        exclude_genes : list, optional
            Genes to exclude from candidates (e.g., disorder genes)
        tolerance : float, optional
            Override default tolerance (number of bins to expand search)

        Returns
        -------
        list
            List of candidate gene identifiers
        """
        if query_gene not in self._binned_genes.index:
            logger.warning(f"Query gene {query_gene} not found in annotations")
            return []

        tolerance = tolerance if tolerance is not None else self.config.tolerance
        exclude_set = set(exclude_genes or [])

        # Get query gene's bins
        query_row = self._binned_genes.loc[query_gene]
        query_length_bin = query_row['length_bin']
        query_cons_bin = query_row['conservation_bin']
        query_expr_bin = query_row['expression_bin']

        candidates = []

        # Start with exact match, then expand if needed
        for tol in range(int(np.ceil(tolerance)) + 1):
            # Generate all bin combinations within tolerance
            length_range = range(
                max(0, int(query_length_bin - tol)),
                min(self.config.length_bins, int(query_length_bin + tol + 1))
            )
            cons_range = range(
                max(0, int(query_cons_bin - tol)),
                min(self.config.conservation_bins, int(query_cons_bin + tol + 1))
            )
            expr_range = range(
                max(0, int(query_expr_bin - tol)),
                min(self.config.expression_bins, int(query_expr_bin + tol + 1))
            )

            for l_bin in length_range:
                for c_bin in cons_range:
                    for e_bin in expr_range:
                        bin_key = f"{l_bin}_{c_bin}_{e_bin}"
                        if bin_key in self._bin_to_genes:
                            for gene in self._bin_to_genes[bin_key]:
                                if gene != query_gene and gene not in exclude_set:
                                    candidates.append(gene)

            # Check if we have enough candidates
            if len(candidates) >= self.config.min_candidates:
                break

        return list(set(candidates))  # Remove duplicates

    def generate_matched_null(
        self,
        query_genes: List,
        gene_weights: Optional[Dict] = None,
        n_permutations: int = 10000,
        mutation_rates: Optional[Dict] = None,
        exclude_genes: Optional[List] = None,
        random_state: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate matched null gene sets for permutation testing.

        Parameters
        ----------
        query_genes : list
            List of query gene identifiers
        gene_weights : dict, optional
            Gene weights for the query set
        n_permutations : int
            Number of null permutations to generate
        mutation_rates : dict, optional
            Gene-level mutation rates for weighted sampling
        exclude_genes : list, optional
            Additional genes to exclude (beyond query genes)
        random_state : int, optional
            Random seed for reproducibility

        Returns
        -------
        pd.DataFrame
            DataFrame with columns [GeneWeight, sim_0, ..., sim_N]
            Each simulation column contains matched null genes
        """
        rng = np.random.default_rng(random_state)

        # Get all genes to exclude (query genes + explicitly excluded)
        all_exclude = set(query_genes)
        if exclude_genes:
            all_exclude.update(exclude_genes)

        # Build candidate pools for each query gene
        logger.info(f"Building candidate pools for {len(query_genes)} genes...")
        gene_candidates: Dict[Union[int, str], List] = {}
        gene_probs: Dict[Union[int, str], np.ndarray] = {}

        for gene in query_genes:
            candidates = self.get_matched_candidates(gene, exclude_genes=list(all_exclude))

            if len(candidates) < 10:
                logger.warning(
                    f"Gene {gene}: only {len(candidates)} candidates. "
                    "Consider relaxing matching stringency."
                )
                # Expand tolerance for this gene
                candidates = self.get_matched_candidates(
                    gene,
                    exclude_genes=list(all_exclude),
                    tolerance=self.config.tolerance + 2
                )

            gene_candidates[gene] = candidates

            # Compute sampling probabilities
            if mutation_rates and self.config.weighted_sampling:
                rates = np.array([mutation_rates.get(c, 1.0) for c in candidates])
                rates = rates / rates.sum()  # Normalize
                gene_probs[gene] = rates
            else:
                gene_probs[gene] = None

        # Generate null permutations
        logger.info(f"Generating {n_permutations} matched null permutations...")
        sim_matrix = np.empty((len(query_genes), n_permutations), dtype=object)

        for i in range(n_permutations):
            used_genes = set()  # Track used genes within this permutation

            for j, gene in enumerate(query_genes):
                candidates = gene_candidates[gene]
                probs = gene_probs[gene]

                # Filter out already-used genes
                available = [c for c in candidates if c not in used_genes]

                if len(available) == 0:
                    # Fallback: allow reuse if no candidates available
                    available = candidates
                    logger.debug(f"Permutation {i}: No unique candidates for gene {gene}")

                # Recompute probabilities for available candidates
                if probs is not None:
                    avail_indices = [candidates.index(c) for c in available]
                    avail_probs = probs[avail_indices]
                    avail_probs = avail_probs / avail_probs.sum()
                else:
                    avail_probs = None

                # Sample one gene
                selected = rng.choice(available, size=1, p=avail_probs, replace=False)[0]
                sim_matrix[j, i] = selected
                used_genes.add(selected)

        # Build output DataFrame
        weights = [gene_weights.get(g, 1.0) if gene_weights else 1.0 for g in query_genes]
        out_df = pd.DataFrame(
            sim_matrix,
            index=query_genes,
            columns=[str(i) for i in range(n_permutations)]
        )
        out_df.insert(0, "GeneWeight", weights)

        return out_df

    def get_matching_statistics(self, query_genes: List) -> pd.DataFrame:
        """
        Get statistics about matching quality for query genes.

        Returns DataFrame with columns:
        - gene: Query gene ID
        - n_candidates: Number of matched candidates
        - length_bin, conservation_bin, expression_bin: Assigned bins
        - mean_candidate_length_diff: Average length difference to candidates
        """
        stats = []

        for gene in query_genes:
            if gene not in self._binned_genes.index:
                continue

            row = self._binned_genes.loc[gene]
            candidates = self.get_matched_candidates(gene)

            stat = {
                'gene': gene,
                'n_candidates': len(candidates),
                'length_bin': row['length_bin'],
                'conservation_bin': row['conservation_bin'],
                'expression_bin': row['expression_bin'],
            }

            if len(candidates) > 0 and 'log_length' in self._binned_genes.columns:
                query_length = row.get('log_length', 0)
                cand_lengths = self._binned_genes.loc[candidates, 'log_length']
                stat['mean_length_diff'] = np.abs(cand_lengths - query_length).mean()

            stats.append(stat)

        return pd.DataFrame(stats)


def load_gene_annotations(
    gnomad_path: Optional[str] = None,
    cds_length_path: Optional[str] = None,
    mutation_rate_path: Optional[str] = None,
    phastcons_path: Optional[str] = None,
    hgnc_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load and merge gene annotations from multiple sources.

    Parameters
    ----------
    gnomad_path : str, optional
        Path to gnomAD constraint metrics file
    cds_length_path : str, optional
        Path to CDS length file
    mutation_rate_path : str, optional
        Path to mutation rate file
    phastcons_path : str, optional
        Path to phastCons conservation file
    hgnc_path : str, optional
        Path to HGNC gene info file

    Returns
    -------
    pd.DataFrame
        Merged gene annotations indexed by entrez_id
    """
    dfs = []

    # Load gnomAD constraint metrics (includes CDS length, pLI, LOEUF)
    if gnomad_path and os.path.exists(gnomad_path):
        logger.info(f"Loading gnomAD annotations from {gnomad_path}")
        gnomad = pd.read_csv(gnomad_path, sep='\t', low_memory=False)
        gnomad = gnomad[['gene', 'cds_length', 'pLI', 'oe_lof_upper', 'brain_expression']].copy()
        gnomad = gnomad.rename(columns={
            'gene': 'gene_symbol',
            'oe_lof_upper': 'LOEUF'
        })
        gnomad = gnomad.drop_duplicates(subset=['gene_symbol'])
        dfs.append(('gnomad', gnomad, 'gene_symbol'))

    # Load CDS length from gencode
    if cds_length_path and os.path.exists(cds_length_path):
        logger.info(f"Loading CDS lengths from {cds_length_path}")
        cds = pd.read_csv(cds_length_path, sep='\t')
        cds = cds[['gene_name', 'cds_len']].copy()
        cds = cds.rename(columns={'gene_name': 'gene_symbol', 'cds_len': 'cds_length_gencode'})
        cds = cds.drop_duplicates(subset=['gene_symbol'])
        dfs.append(('cds', cds, 'gene_symbol'))

    # Load mutation rates
    if mutation_rate_path and os.path.exists(mutation_rate_path):
        logger.info(f"Loading mutation rates from {mutation_rate_path}")
        mr = pd.read_csv(mutation_rate_path, sep='\t', low_memory=False)
        # Use relevant columns
        mr_cols = ['GeneName', 'p_LGD', 'prevel_0.5', 'p_misense']
        mr_available = [c for c in mr_cols if c in mr.columns]
        if 'GeneName' in mr_available:
            mr = mr[mr_available].copy()
            mr = mr.rename(columns={'GeneName': 'gene_symbol'})
            mr = mr.drop_duplicates(subset=['gene_symbol'])
            dfs.append(('mutation_rate', mr, 'gene_symbol'))

    # Load HGNC for entrez ID mapping
    if hgnc_path and os.path.exists(hgnc_path):
        logger.info(f"Loading HGNC gene info from {hgnc_path}")
        hgnc = pd.read_csv(hgnc_path, sep='\t', low_memory=False)
        hgnc = hgnc[['symbol', 'entrez_id', 'ensembl_gene_id']].copy()
        hgnc['entrez_id'] = pd.to_numeric(hgnc['entrez_id'], errors='coerce')
        hgnc = hgnc.dropna(subset=['entrez_id'])
        hgnc['entrez_id'] = hgnc['entrez_id'].astype(int)
        hgnc = hgnc.rename(columns={'symbol': 'gene_symbol'})
        hgnc = hgnc.drop_duplicates(subset=['gene_symbol'])
        dfs.append(('hgnc', hgnc, 'gene_symbol'))

    if not dfs:
        raise ValueError("No annotation files provided or found")

    # Merge all DataFrames on gene_symbol
    result = dfs[0][1].copy()
    for name, df, key in dfs[1:]:
        result = result.merge(df, on=key, how='outer')

    # Fill in CDS length from gencode if missing from gnomAD
    if 'cds_length' in result.columns and 'cds_length_gencode' in result.columns:
        result['cds_length'] = result['cds_length'].fillna(result['cds_length_gencode'])
        result = result.drop(columns=['cds_length_gencode'])

    # Set index to entrez_id if available
    if 'entrez_id' in result.columns:
        result = result.dropna(subset=['entrez_id'])
        result['entrez_id'] = result['entrez_id'].astype(int)
        result = result.set_index('entrez_id')

    logger.info(f"Loaded annotations for {len(result)} genes")
    return result


def compute_phastcons_per_gene(
    bw_path: str,
    gene_coords_path: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute mean phastCons conservation scores per gene.

    This function requires pyBigWig to be installed.

    Parameters
    ----------
    bw_path : str
        Path to phastCons bigWig file (e.g., hg19.100way.phastCons.bw)
    gene_coords_path : str
        Path to gene coordinates file (gene, chr, start, end)
    output_path : str, optional
        Path to save computed scores

    Returns
    -------
    pd.DataFrame
        DataFrame with gene and phastcons columns
    """
    try:
        import pyBigWig
    except ImportError:
        raise ImportError(
            "pyBigWig is required for computing phastCons scores. "
            "Install with: pip install pyBigWig"
        )

    logger.info(f"Computing phastCons scores from {bw_path}")

    # Load gene coordinates
    coords = pd.read_csv(gene_coords_path, sep='\t')

    bw = pyBigWig.open(bw_path)

    scores = []
    for _, row in coords.iterrows():
        chrom = row['chr'] if row['chr'].startswith('chr') else f"chr{row['chr']}"
        start = int(row['start'])
        end = int(row['end'])

        try:
            vals = bw.values(chrom, start, end)
            mean_score = np.nanmean(vals) if vals else np.nan
        except Exception:
            mean_score = np.nan

        scores.append({
            'gene': row['gene'],
            'phastcons': mean_score
        })

    bw.close()

    result = pd.DataFrame(scores)

    if output_path:
        result.to_csv(output_path, index=False)
        logger.info(f"Saved phastCons scores to {output_path}")

    return result


def run_sensitivity_analysis(
    query_genes: List,
    gene_weights: Dict,
    gene_annotations: pd.DataFrame,
    expression_matrix: pd.DataFrame,
    stringency_levels: List[str] = ['loose', 'medium', 'tight', 'very_tight'],
    n_permutations: int = 1000,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Run sensitivity analysis across different matching stringencies.

    Parameters
    ----------
    query_genes : list
        List of query gene identifiers
    gene_weights : dict
        Gene weights dictionary
    gene_annotations : pd.DataFrame
        Gene annotation DataFrame
    expression_matrix : pd.DataFrame
        Expression matrix
    stringency_levels : list
        List of stringency levels to test
    n_permutations : int
        Number of permutations per level
    random_state : int
        Random seed

    Returns
    -------
    dict
        Dictionary mapping stringency level to null gene weights DataFrame
    """
    # Define stringency configurations
    stringency_configs = {
        'loose': MatchingConfig(
            length_bins=5, conservation_bins=5, expression_bins=5,
            tolerance=2.0, min_candidates=100
        ),
        'medium': MatchingConfig(
            length_bins=10, conservation_bins=10, expression_bins=10,
            tolerance=1.0, min_candidates=50
        ),
        'tight': MatchingConfig(
            length_bins=15, conservation_bins=15, expression_bins=15,
            tolerance=0.5, min_candidates=30
        ),
        'very_tight': MatchingConfig(
            length_bins=20, conservation_bins=20, expression_bins=20,
            tolerance=0.0, min_candidates=20
        )
    }

    results = {}

    for level in stringency_levels:
        if level not in stringency_configs:
            logger.warning(f"Unknown stringency level: {level}")
            continue

        logger.info(f"Running {level} stringency matching...")

        config = stringency_configs[level]
        matcher = GenePropertyMatcher(
            gene_annotations=gene_annotations,
            expression_matrix=expression_matrix,
            config=config
        )

        null_df = matcher.generate_matched_null(
            query_genes=query_genes,
            gene_weights=gene_weights,
            n_permutations=n_permutations,
            random_state=random_state
        )

        results[level] = null_df

        # Log matching quality
        stats = matcher.get_matching_statistics(query_genes)
        mean_candidates = stats['n_candidates'].mean()
        logger.info(f"  {level}: mean {mean_candidates:.1f} candidates per gene")

    return results
