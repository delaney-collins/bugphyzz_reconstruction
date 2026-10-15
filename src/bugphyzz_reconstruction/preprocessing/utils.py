"""Utility functions translated from taxPPro's utils.R.

The functions in this module convert Bugphyzz frequency labels, confidence
levels, taxonomy ranks, and NSTI values into the formats required by the ASR
pipeline.

The implementations use vectorized pandas and NumPy operations rather than
row-by-row loops.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_RANKS: tuple[str, ...] = (
    "strain",
    "species",
    "genus",
    "family",
    "order",
    "class",
    "phylum",
    "superkingdom",
)

VALID_PARENT_RANKS: tuple[str, ...] = (
    "species",
    "genus",
    "family",
)

CONFIDENCE_LEVELS: tuple[str, ...] = (
    "low",
    "medium",
    "high",
)

FREQUENCY_TO_SCORE: dict[str, float] = {
    "always": 1.0,
    "usually": 0.9,
    "sometimes": 0.5,
    "unknown": 0.1,
}

POST_ASR_FREQUENCY_TO_SCORE: dict[str, float] = {
    "always": 1.0,
    "usually": 0.9,
    "sometimes": 0.5,
    "rarely": 0.1,
    "never": 0.0,
}

RANK_PREFIXES: dict[str, str] = {
    "kingdom": "k__",
    "superkingdom": "k__",
    "domain": "k__",
    "phylum": "p__",
    "class": "c__",
    "order": "o__",
    "family": "f__",
    "genus": "g__",
    "species": "s__",
    "strain": "t__",
}


# ---------------------------------------------------------------------------
# Internal conversion helper
# ---------------------------------------------------------------------------

def _as_series(
    values: pd.Series | Sequence[Any] | np.ndarray,
    *,
    name: str | None = None,
) -> pd.Series:
    """Convert array-like input to a pandas Series.

    Existing Series objects retain their original index. Other array-like
    objects receive a default RangeIndex.
    """
    if isinstance(values, pd.Series):
        result = values.copy()

        if name is not None:
            result.name = name

        return result

    return pd.Series(values, name=name)


# ---------------------------------------------------------------------------
# Frequency and score conversion
# ---------------------------------------------------------------------------

def frequency_to_score(
    values: pd.Series | Sequence[Any] | np.ndarray,
) -> pd.Series:
    """Convert Bugphyzz frequency labels to numeric probability scores.

    Equivalent to the R function ``freq2Scores``.

    Mapping
    -------
    always
        1.0
    usually
        0.9
    sometimes
        0.5
    unknown
        0.1

    Unrecognized or missing values become ``NaN``.

    Parameters
    ----------
    values
        Frequency labels.

    Returns
    -------
    pandas.Series
        Nullable floating-point scores.
    """
    series = _as_series(values)

    normalized = series.astype("string").str.lower()

    return normalized.map(FREQUENCY_TO_SCORE).astype("Float64")


def frequency_to_post_asr_score(
    values: pd.Series | Sequence[Any] | np.ndarray,
) -> pd.Series:
    """Convert post-ASR frequency labels to numeric probability scores.

    Equivalent to the R function ``freq2Scores2``.

    This version is used after ASR for numeric attributes.

    Mapping
    -------
    always
        1.0
    usually
        0.9
    sometimes
        0.5
    rarely
        0.1
    never
        0.0

    Unrecognized or missing values become ``NaN``.
    """
    series = _as_series(values)

    normalized = series.astype("string").str.lower()

    return normalized.map(POST_ASR_FREQUENCY_TO_SCORE).astype("Float64")

def score_to_frequency(
    values: pd.Series,
) -> pd.Series:
    """Convert scores to canonical Bugphyzz frequency labels."""
    scores = pd.to_numeric(values, 
                           errors="coerce"
                           ).astype("float64")

    conditions = [
        scores.eq(1.0),
        scores.ge(0.9) & scores.lt(1.0),
        scores.ge(0.5) & scores.lt(0.9),
        scores.gt(0.0) & scores.lt(0.5),
        scores.eq(0.0),
    ]

    labels = [
        "always",
        "usually",
        "sometimes",
        "rarely",
        "never",
    ]

    return pd.Series(
        pd.array(
            np.select(conditions, labels, default=None),
            dtype="string",
        ),
        index=scores.index,
        name=values.name,
    )

# ---------------------------------------------------------------------------
# Confidence conversion
# ---------------------------------------------------------------------------

def confidence_to_categorical(
    values: pd.Series | Sequence[Any] | np.ndarray,
) -> pd.Series:
    """Convert curation confidence labels to an ordered categorical Series.

    Equivalent to the R function ``conf2Fct``.

    The ordering is:

    ``low < medium < high``

    Unrecognized values become missing categorical values.
    """
    series = _as_series(values)

    categorical = pd.Categorical(
        series,
        categories=list(CONFIDENCE_LEVELS),
        ordered=True,
    )

    return pd.Series(
        categorical,
        index=series.index,
        name=series.name,
    )


# ---------------------------------------------------------------------------
# Taxonomic rank helpers
# ---------------------------------------------------------------------------

def valid_ranks() -> tuple[str, ...]:
    """Return the valid taxonomic ranks used by taxPPro."""
    return VALID_RANKS


def valid_parent_ranks() -> tuple[str, ...]:
    """Return the valid parent ranks used by taxPPro."""
    return VALID_PARENT_RANKS


def add_rank_prefix(
    ids: pd.Series | Sequence[Any] | np.ndarray,
    ranks: pd.Series | Sequence[Any] | np.ndarray,
) -> pd.Series:
    """Add rank prefixes to NCBI taxonomy IDs.

    Equivalent to the R function ``addRankPrefix``.

    Prefix mapping
    --------------
    kingdom, superkingdom, domain
        ``k__``
    phylum
        ``p__``
    class
        ``c__``
    order
        ``o__``
    family
        ``f__``
    genus
        ``g__``
    species
        ``s__``
    strain
        ``t__``

    Unknown ranks or missing IDs return ``pd.NA``.

    Parameters
    ----------
    ids
        NCBI taxonomy IDs.
    ranks
        Taxonomic ranks. Must have the same length as ``ids``.

    Returns
    -------
    pandas.Series
        Rank-prefixed IDs.
    """
    id_series = _as_series(ids, name="NCBI_ID")
    rank_series = _as_series(ranks, name="Rank")

    if len(id_series) != len(rank_series):
        raise ValueError(
            "ids and ranks must contain the same number of values."
        )

    # Align by position, matching R's vectorized behavior.
    rank_series.index = id_series.index

    normalized_ranks = rank_series.astype("string").str.lower()
    prefixes = normalized_ranks.map(RANK_PREFIXES)

    id_strings = id_series.astype("string")

    valid = prefixes.notna() & id_strings.notna()

    result = pd.Series(
        pd.NA,
        index=id_series.index,
        dtype="string",
        name=id_series.name,
    )

    result.loc[valid] = (
        prefixes.loc[valid].astype("string")
        + id_strings.loc[valid]
    )

    return result


# ---------------------------------------------------------------------------
# Taxonomy classification reshaping
# ---------------------------------------------------------------------------

def classification_to_table(
    classification: pd.DataFrame,
    ranks: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Convert a taxize-style classification table to one wide row.

    Equivalent to the R function ``classif2Table``.

    The input must contain:

    - ``rank``
    - ``id``

    For example, this input:

    +----------------+--------+
    | rank           | id     |
    +================+========+
    | superkingdom   | 2      |
    | phylum         | 1224   |
    | genus          | 561    |
    | species        | 562    |
    +----------------+--------+

    becomes:

    +----------------+--------+-------+---------+
    | superkingdom   | phylum | genus | species |
    +================+========+=======+=========+
    | 2              | 1224   | 561   | 562     |
    +----------------+--------+-------+---------+

    Parameters
    ----------
    classification
        Taxonomy classification table.
    ranks
        Ranks to retain. Defaults to :data:`VALID_RANKS`.

    Returns
    -------
    pandas.DataFrame
        A one-row table whose columns are taxonomic ranks.

    Raises
    ------
    KeyError
        If ``classification`` does not contain ``rank`` and ``id``.
    ValueError
        If more than one ID is present for the same retained rank.
    """
    required_columns = {"rank", "id"}
    missing_columns = required_columns.difference(classification.columns)

    if missing_columns:
        raise KeyError(
            "classification is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    selected_ranks = tuple(ranks) if ranks is not None else VALID_RANKS

    filtered = classification.loc[
        classification["rank"].isin(selected_ranks),
        ["rank", "id"],
    ].copy()

    if filtered.empty:
        return pd.DataFrame()

    duplicated_ranks = filtered["rank"].duplicated(keep=False)

    if duplicated_ranks.any():
        duplicate_names = (
            filtered.loc[duplicated_ranks, "rank"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Multiple taxonomy IDs were found for the same rank: "
            + ", ".join(duplicate_names)
        )

    result = filtered.set_index("rank")["id"].to_frame().T
    result.index = pd.RangeIndex(start=0, stop=len(result))

    # Preserve the order found in the input, matching the R implementation.
    return result


# ---------------------------------------------------------------------------
# Probability threshold
# ---------------------------------------------------------------------------

def round_probability_threshold(
    values: pd.Series | Sequence[Any] | np.ndarray,
) -> float:
    """Calculate the next tenth above the reciprocal state count.

    Equivalent to the R function ``roundDec``.

    The function counts unique, non-missing values and calculates:

    ``ceil(((1 / n) + 1e-6) * 10) / 10``

    This produces a threshold slightly above the equal-probability value for
    ``n`` possible states.

    Examples
    --------
    Two states:

    ``1 / 2 = 0.5`` becomes ``0.6``.

    Three states:

    ``1 / 3 = 0.333...`` becomes ``0.4``.

    Parameters
    ----------
    values
        Attribute or state labels.

    Returns
    -------
    float
        Probability threshold.

    Raises
    ------
    ValueError
        If there are no unique non-missing values.
    """
    series = _as_series(values)

    n_unique = series.dropna().nunique()

    if n_unique == 0:
        raise ValueError(
            "Cannot calculate a probability threshold because no "
            "non-missing values were provided."
        )

    return float(
        np.ceil(((1.0 / n_unique) + 1e-6) * 10.0) / 10.0
    )


# ---------------------------------------------------------------------------
# NSTI conversion
# ---------------------------------------------------------------------------

def nsti_to_frequency(
    values: pd.Series | Sequence[Any] | np.ndarray,
) -> pd.Series:
    """Convert NSTI values to Bugphyzz frequency labels.

    Equivalent to the R function ``nsti2Freq``.

    Classification
    --------------
    missing
        missing
    x == 0
        ``always``
    0 < x <= 0.05
        ``usually``
    0.05 < x <= 0.15
        ``sometimes``
    0.15 < x <= 0.20
        ``rarely``
    x > 0.20
        ``never``

    Negative NSTI values are not classified and become missing.
    """
    series = pd.to_numeric(
        _as_series(values),
        errors="coerce",
    )

    conditions = [
        series.eq(0.0),
        series.gt(0.0) & series.le(0.05),
        series.gt(0.05) & series.le(0.15),
        series.gt(0.15) & series.le(0.20),
        series.gt(0.20),
    ]

    choices = [
        "always",
        "usually",
        "sometimes",
        "rarely",
        "never",
    ]

    result = np.select(
        conditions,
        choices,
        default=None,
    )

    return pd.Series(
        pd.array(result, dtype="string"),
        index=series.index,
        name=series.name,
    )


# ---------------------------------------------------------------------------
# Aliases matching the original R names
# ---------------------------------------------------------------------------

# These aliases can make comparisons with the R pipeline easier while the
# Python port is being validated. The snake_case names above are preferable
# for normal Python use.

freq2Scores = frequency_to_score
freq2Scores2 = frequency_to_post_asr_score
scores2Freq = score_to_frequency
conf2Fct = confidence_to_categorical
addRankPrefix = add_rank_prefix
classif2Table = classification_to_table
roundDec = round_probability_threshold
nsti2Freq = nsti_to_frequency

