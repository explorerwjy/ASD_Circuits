"""
Shared test fixtures for GENCIC webapp tests.

Provides synthetic data (small matrices) for fast unit tests and
optional access to real data for integration tests.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure src/ and webapp/ are importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Real data paths (may not exist in worktree)
# ---------------------------------------------------------------------------
_REAL_DATA_ROOT = Path("/home/jw3514/Work/ASD_Circuits_CellType/dat")
REAL_DATA_AVAILABLE = (
    (_REAL_DATA_ROOT / "allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv").exists()
    and (_REAL_DATA_ROOT / "BiasMatrices/AllenMouseBrain_Z2bias.parquet").exists()
)

skip_without_real_data = pytest.mark.skipif(
    not REAL_DATA_AVAILABLE,
    reason="Real data files not available in worktree",
)


# ---------------------------------------------------------------------------
# Synthetic small matrices for unit tests
# ---------------------------------------------------------------------------

N_STRUCTURES = 30  # small for fast tests


@pytest.fixture
def rng():
    """Seeded NumPy random generator."""
    return np.random.default_rng(42)


@pytest.fixture
def structure_names():
    """Fake structure names."""
    return np.array([f"STR_{i:03d}" for i in range(N_STRUCTURES)])


@pytest.fixture
def synthetic_info_mat(structure_names, rng):
    """Symmetric (30 x 30) information matrix with some zero entries."""
    n = len(structure_names)
    raw = rng.uniform(0, 1, size=(n, n))
    # Make it sparse-ish: set ~30% to zero
    mask = rng.random(size=(n, n)) < 0.3
    raw[mask] = 0.0
    # Ensure diagonal is nonzero (self-connections)
    np.fill_diagonal(raw, rng.uniform(0.5, 1.0, size=n))
    return pd.DataFrame(raw, index=structure_names, columns=structure_names)


@pytest.fixture
def synthetic_adj_mat(structure_names, rng):
    """Symmetric (30 x 30) adjacency weight matrix."""
    n = len(structure_names)
    raw = rng.uniform(0, 1, size=(n, n))
    # Make symmetric
    raw = (raw + raw.T) / 2
    mask = rng.random(size=(n, n)) < 0.4
    raw[mask] = 0.0
    return pd.DataFrame(raw, index=structure_names, columns=structure_names)


@pytest.fixture
def synthetic_bias_df(structure_names, rng):
    """Synthetic bias DataFrame sorted by EFFECT descending.

    Columns: EFFECT, Rank, REGION.  Index = structure name.
    """
    effects = np.sort(rng.uniform(0.0, 0.5, size=len(structure_names)))[::-1]
    df = pd.DataFrame(
        {
            "EFFECT": effects,
            "Rank": np.arange(1, len(structure_names) + 1),
            "REGION": "TestRegion",
        },
        index=structure_names,
    )
    return df


# ---------------------------------------------------------------------------
# Real data fixtures (guarded by skip_without_real_data)
# ---------------------------------------------------------------------------

@pytest.fixture
def real_info_mat():
    """Load real InfoMat.Ipsi.csv (213 x 213)."""
    path = _REAL_DATA_ROOT / "allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv"
    return pd.read_csv(path, index_col=0)


@pytest.fixture
def real_adj_mat():
    """Load real WeightMat.Ipsi.csv (213 x 213)."""
    path = _REAL_DATA_ROOT / "allen-mouse-conn/ConnectomeScoringMat/WeightMat.Ipsi.csv"
    return pd.read_csv(path, index_col=0)


@pytest.fixture
def real_str_bias_matrix():
    """Load real Allen Mouse Brain Z2 bias parquet."""
    path = _REAL_DATA_ROOT / "BiasMatrices/AllenMouseBrain_Z2bias.parquet"
    return pd.read_parquet(path)
