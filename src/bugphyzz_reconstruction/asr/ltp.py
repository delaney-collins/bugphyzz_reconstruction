"""Living Tree Project resources and NSTI calculations.

Python translation of the taxPPro functions:

- ltp()
- getNsti()

Dependencies
------------
pandas
biopython
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
import math
import numbers
from pathlib import Path
import sys
from typing import Collection, Iterable, Sequence

import numpy as np
import pandas as pd
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree


TREE_FILENAME = "LTP_all_08_2023.newick"
TIP_DATA_FILENAME = "LTP_all_08_2023.tip_data"
NODE_DATA_FILENAME = "LTP_all_08_2023.node_data"


@dataclass(frozen=True)
class LTPResources:
    """Container corresponding to the list returned by the R ltp() function."""

    tree: Tree
    tip_data: pd.DataFrame
    node_data: pd.DataFrame
    gn_tips: list[str]


@dataclass(frozen=True)
class HSPResult:
    """Result corresponding to castor's HSP return list.

    ``states`` follows ape/castor order: all terminal tips in tree order,
    followed by all internal nodes in root-first preorder.
    """

    success: bool
    total_sum_of_squared_changes: float
    states: np.ndarray


def _default_resource_dir() -> Path:
    """Return the package's default resources directory.

    This assumes this module is located directly inside the bugphyzz reconstruction package:

        bugphyzz_reconstruction/
        ├── ltp.py
        └── resources/taxonomy_2023
            ├── LTP_all_08_2023.newick
            ├── LTP_all_08_2023.tip_data
            └── LTP_all_08_2023.node_data
    """
    return Path(__file__).resolve().parents[1] / "resources"/"taxonomy_2023"


def _read_character_table(path: Path) -> pd.DataFrame:
    """Read a tab-delimited table while retaining character-like columns.

    This approximates:

        read.table(..., header = TRUE, sep = "\\t") |>
            purrr::modify(as.character) |>
            as.data.frame()

    Missing values remain pandas NA values rather than becoming the literal
    string "NA".
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=True,
        na_values=["NA"],
    )


def _tip_labels(tree: Tree) -> list[str]:
    """Return terminal labels in Newick/tree order."""
    labels: list[str] = []

    for tip in tree.get_terminals(order="preorder"):
        if tip.name is None:
            raise ValueError("The LTP tree contains an unnamed terminal tip.")

        labels.append(str(tip.name))

    return labels


def _node_clades(tree: Tree) -> list[Clade]:
    """Return internal nodes in root-first preorder.

    ape numbers tips from 1 through Ntip and internal nodes beginning at
    Ntip + 1. For a normally read Newick tree, root-first preorder reproduces
    the corresponding root-first internal-node ordering.
    """
    return list(tree.get_nonterminals(order="preorder"))


def _add_missing_node_names(tree: Tree) -> None:
    """Replace internal-node label 'NA' using ape-style node numbers.

    This translates:

        tree$node.label <- ifelse(
            tree$node.label == "NA",
            paste0("n", Ntip(tree) + 1:Nnode(tree)),
            tree$node.label
        )

    Only the literal label ``"NA"`` is replaced. A genuinely absent Newick
    label represented by ``None`` is left unchanged, matching the original
    condition as closely as possible.
    """
    n_tips = len(tree.get_terminals())
    internal_nodes = _node_clades(tree)

    for offset, node in enumerate(internal_nodes, start=1):
        if node.name == "NA":
            node.name = f"n{n_tips + offset}"

def _restore_numeric_internal_node_labels(tree: Tree) -> None:
    """Move numeric Newick node labels from confidence to name.

    Biopython parses an internal label such as:

        )2157:

    as ``clade.confidence = 2157`` rather than ``clade.name = "2157"``.

    R's ape::read.tree() retains these values in tree$node.label, so move
    them back to ``name`` for parity with the original taxPPro workflow.
    """
    for node in tree.get_nonterminals(order="preorder"):
        if node.name is not None or node.confidence is None:
            continue

        confidence = float(node.confidence)

        if confidence.is_integer():
            node.name = str(int(confidence))
        else:
            node.name = format(confidence, ".15g")

        node.confidence = None

def ltp(
    remove_gn_nodes: bool = True,
    node_names: bool = True,
    resource_dir: str | Path | None = None,
) -> LTPResources:
    """Load the modified Living Tree Project tree and metadata.

    Parameters
    ----------
    remove_gn_nodes
        Remove rows from ``node_data`` whose ``Rank`` is ``"genus"``.
        These genera are already represented in ``tip_data``.

    node_names
        Replace internal-node labels equal to the literal string ``"NA"``
        with ``n`` followed by the corresponding ape-style node number.

        The generated labels are added only to the tree. They are not added
        to ``node_data``.

    resource_dir
        Directory containing the three LTP resource files. When omitted,
        ``bugphyzz/resources`` relative to this module is used.

    Returns
    -------
    LTPResources
        Container with:

        - ``tree``
        - ``tip_data``
        - ``node_data``
        - ``gn_tips``
    """
    resources = (
        Path(resource_dir)
        if resource_dir is not None
        else _default_resource_dir()
    )

    tree_path = resources / TREE_FILENAME
    tip_data_path = resources / TIP_DATA_FILENAME
    node_data_path = resources / NODE_DATA_FILENAME

    required_files = (tree_path, tip_data_path, node_data_path)
    missing_files = [path for path in required_files if not path.is_file()]

    if missing_files:
        formatted = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(
            "One or more LTP resource files were not found:\n"
            f"{formatted}"
        )

    tree = Phylo.read(str(tree_path), "newick")

    _restore_numeric_internal_node_labels(tree)

    if node_names:
        _add_missing_node_names(tree)

    tip_data = _read_character_table(tip_data_path)

    if "tip_label" not in tip_data.columns:
        raise ValueError(
            f"{tip_data_path.name} does not contain a 'tip_label' column."
        )

    # Equivalent to:
    # rownames(tip_data) <- tip_data$tip_label
    #
    # drop=False retains tip_label as a normal column.
    tip_data = tip_data.set_index("tip_label", drop=False)

    node_data = _read_character_table(node_data_path)

    tip_labels = tip_data["tip_label"].dropna().astype(str)
    gn_tips = [
        label
        for label in tip_labels
        if "g__" in label
    ]

    if remove_gn_nodes:
        if "Rank" not in node_data.columns:
            raise ValueError(
                f"{node_data_path.name} does not contain a 'Rank' column."
            )

        # dplyr::filter(Rank != "genus") excludes both genus rows and rows
        # where Rank is missing, because NA conditions are not retained.
        keep = node_data["Rank"].notna() & node_data["Rank"].ne("genus")
        node_data = node_data.loc[keep].copy()

        # Equivalent to:
        # purrr::discard(~all(is.na(.x)))
        node_data = node_data.dropna(axis="columns", how="all")

    return LTPResources(
        tree=tree,
        tip_data=tip_data,
        node_data=node_data,
        gn_tips=gn_tips,
    )


def _build_tree_adjacency(
    tree: Tree,
) -> tuple[
    dict[Clade, list[tuple[Clade, float]]],
    dict[str, Clade],
    list[Clade],
]:
    """Represent the rooted tree as an undirected weighted graph."""
    adjacency: dict[Clade, list[tuple[Clade, float]]] = {}
    tip_by_label: dict[str, Clade] = {}

    def add_node(node: Clade) -> None:
        adjacency.setdefault(node, [])

        if node.is_terminal():
            if node.name is None:
                raise ValueError("The LTP tree contains an unnamed tip.")

            label = str(node.name)

            if label in tip_by_label:
                raise ValueError(
                    f"The LTP tree contains duplicate tip label {label!r}."
                )

            tip_by_label[label] = node

        for child in node.clades:
            if child.branch_length is None:
                raise ValueError(
                    "The LTP tree contains an edge without a branch length. "
                    "NSTI requires weighted phylogenetic distances."
                )

            distance = float(child.branch_length)

            if distance < 0:
                raise ValueError(
                    "The LTP tree contains a negative branch length."
                )

            adjacency.setdefault(child, [])
            adjacency[node].append((child, distance))
            adjacency[child].append((node, distance))

            add_node(child)

    add_node(tree.root)

    internal_nodes = _node_clades(tree)

    return adjacency, tip_by_label, internal_nodes


def _nearest_source_distances(
    adjacency: dict[Clade, list[tuple[Clade, float]]],
    sources: Iterable[Clade],
) -> dict[Clade, float]:
    """Calculate distance to the nearest source using multi-source Dijkstra."""
    distances = {
        node: float("inf")
        for node in adjacency
    }

    queue: list[tuple[float, int, Clade]] = []
    counter = 0

    for source in sources:
        if distances[source] == 0.0:
            continue

        distances[source] = 0.0
        heappush(queue, (0.0, counter, source))
        counter += 1

    while queue:
        current_distance, _, node = heappop(queue)

        if current_distance != distances[node]:
            continue

        for neighbor, edge_length in adjacency[node]:
            candidate = current_distance + edge_length

            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heappush(queue, (candidate, counter, neighbor))
                counter += 1

    return distances


def get_nsti(
    tree: Tree,
    annotated_tip_labels: Collection[str],
    nodes: Collection[str],
) -> pd.DataFrame:
    """Calculate nearest-sequenced-taxon-index distances.

    This translates the R function:

        getNsti(tree, annotated_tip_labels, nodes)

    Distances are calculated from every tree vertex to its nearest annotated
    tip. The returned table contains:

    - unannotated terminal tips
    - internal nodes whose labels occur in ``nodes``

    Parameters
    ----------
    tree
        Parsed LTP tree returned by :func:`ltp`.

    annotated_tip_labels
        Tip labels with observed annotations for the current continuous trait.

    nodes
        Internal-node labels that should be retained. In the original
        workflow, these are the tree nodes associated with taxonomic IDs.

    Returns
    -------
    pandas.DataFrame
        Columns:

        - ``label``
        - ``nsti``

        This follows the actual R implementation. Although the Roxygen text
        mentions ``tip_label``, the function itself returns ``label``.
    """
    adjacency, tip_by_label, internal_nodes = _build_tree_adjacency(tree)

    annotated_set = {
        str(label)
        for label in annotated_tip_labels
        if not pd.isna(label)
    }

    node_label_set = {
        str(label)
        for label in nodes
        if not pd.isna(label)
    }

    tree_tip_labels = set(tip_by_label)

    missing_annotated_labels = annotated_set - tree_tip_labels
    if missing_annotated_labels:
        preview = sorted(missing_annotated_labels)[:10]
        raise ValueError(
            f"{len(missing_annotated_labels)} annotated tip label(s) are "
            "absent from the LTP tree. First values: "
            f"{preview}"
        )

    if not annotated_set:
        raise ValueError(
            "At least one annotated tree tip is required to calculate NSTI."
        )

    annotated_tips = [
        tip_by_label[label]
        for label in annotated_set
    ]

    nearest_distances = _nearest_source_distances(
        adjacency=adjacency,
        sources=annotated_tips,
    )

    # Preserve the original tree tip order.
    tip_rows = [
        {
            "label": label,
            "nsti": nearest_distances[tip],
        }
        for label, tip in tip_by_label.items()
        if label not in annotated_set
    ]

    # Preserve internal-node order and then filter by the supplied labels.
    node_rows: list[dict[str, str | float]] = []

    for node in internal_nodes:
        if node.name is None:
            continue

        label = str(node.name)

        if label in node_label_set:
            node_rows.append(
                {
                    "label": label,
                    "nsti": nearest_distances[node],
                }
            )

    return pd.DataFrame(
        tip_rows + node_rows,
        columns=["label", "nsti"],
    )


def _edge_weight(
    child: Clade,
    *,
    weighted: bool,
) -> float:
    """Return the squared-change penalty associated with one edge."""
    if not weighted or child.branch_length is None:
        return 1.0

    length = float(child.branch_length)
    if not math.isfinite(length):
        raise ValueError("The tree contains a non-finite branch length.")
    if length < 0:
        raise ValueError("The tree contains a negative branch length.")

    # castor permits zero-length edges and internally substitutes a tiny
    # positive length when weighted=TRUE.
    return 1.0 / max(length, sys.float_info.epsilon)


def _length_weight(length: float | None, *, weighted: bool) -> float:
    """Return a penalty for an original or collapsed edge length."""
    if not weighted or length is None:
        return 1.0
    if not math.isfinite(length):
        raise ValueError("The tree contains a non-finite branch length.")
    if length < 0:
        raise ValueError("The tree contains a negative branch length.")
    return 1.0 / max(length, sys.float_info.epsilon)


def _coerce_tip_states(
    tree: Tree,
    tip_states: Sequence[float | int | None] | pd.Series | np.ndarray,
    *,
    check_input: bool,
) -> tuple[list[Clade], np.ndarray]:
    """Validate and convert a castor-style tip-state vector."""
    tips = list(tree.get_terminals(order="preorder"))
    tip_labels = [
        str(tip.name)
        if tip.name is not None
        else None
        for tip in tips
    ]

    if isinstance(tip_states, pd.Series):
        if check_input and not isinstance(tip_states.index, pd.RangeIndex):
            supplied_labels = [str(label) for label in tip_states.index]
            if supplied_labels != tip_labels:
                raise ValueError(
                    "Names in tip_states and tip labels in tree do not "
                    "match in the same order."
                )
        raw_states = tip_states.tolist()
    else:
        if isinstance(tip_states, (str, bytes)):
            raise TypeError("tip_states must be a numeric one-dimensional vector.")
        raw_states = list(tip_states)

    if len(raw_states) != len(tips):
        raise ValueError(
            f"Length of tip_states ({len(raw_states)}) is not the same as "
            f"the number of tips in the tree ({len(tips)})."
        )

    values = np.empty(len(raw_states), dtype=float)
    for index, value in enumerate(raw_states):
        if value is None or pd.isna(value):
            values[index] = np.nan
            continue

        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, numbers.Real
        ):
            raise TypeError("tip_states must be a numeric vector.")

        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(
                "Known tip states must be finite numeric values or missing."
            )
        values[index] = numeric_value

    if np.isnan(values).all():
        raise ValueError("All tip states are hidden.")

    return tips, values


def hsp_squared_change_parsimony(
    tree: Tree,
    tip_states: Sequence[float | int | None] | pd.Series | np.ndarray,
    weighted: bool = True,
    check_input: bool = True,
) -> HSPResult:
    """Predict hidden continuous states using squared-change parsimony.

    This is a linear-time Python translation of
    ``castor::hsp_squared_change_parsimony``. It minimizes

    .. math::

        \\sum_{(u,v)} w_{uv}(x_u-x_v)^2

    while holding observed tip values fixed. With ``weighted=True``,
    ``w`` is the inverse branch length; otherwise every edge has unit
    weight.

    Hidden terminal clades do not contribute to the objective and are
    forward-filled with the state of their nearest reconstructed ancestor,
    matching castor's post-reconstruction projection behavior.

    Parameters
    ----------
    tree
        A rooted Biopython tree.
    tip_states
        One value per terminal tip in preorder. Missing values may be
        represented by ``None``, ``NaN``, or ``pandas.NA``. A named pandas
        Series is checked against tree tip labels when ``check_input=True``.
    weighted
        Weight squared changes by inverse branch length.
    check_input
        Check named pandas Series indices against tree tip order.

    Returns
    -------
    HSPResult
        ``success``, minimized ``total_sum_of_squared_changes``, and states
        in tip-then-internal-node order.
    """
    tips, values = _coerce_tip_states(
        tree,
        tip_states,
        check_input=check_input,
    )
    tip_value = {
        tip: values[index]
        for index, tip in enumerate(tips)
        if not np.isnan(values[index])
    }

    # Reproduce castor::get_subtree_with_tips(...,
    # collapse_monofurcations=TRUE, force_keep_root=TRUE). ``active`` marks
    # clades on a path from the root to at least one observed tip. Unary
    # active internal nodes are then omitted from the reconstruction tree.
    active: dict[Clade, bool] = {}
    active_child_count: dict[Clade, int] = {}

    for node in tree.find_clades(order="postorder"):
        child_count = sum(active.get(child, False) for child in node.clades)
        active_child_count[node] = child_count
        active[node] = node in tip_value or child_count > 0

    retained = {
        node
        for node in tree.find_clades(order="preorder")
        if (
            node is tree.root
            or node in tip_value
            or (active[node] and active_child_count[node] != 1)
        )
    }

    retained_children: dict[Clade, list[tuple[Clade, float | None]]] = {
        node: []
        for node in retained
    }

    for parent in retained:
        for first_child in parent.clades:
            if not active[first_child]:
                continue

            child = first_child
            path_length = child.branch_length

            while child not in retained:
                active_children = [
                    descendant
                    for descendant in child.clades
                    if active[descendant]
                ]
                if len(active_children) != 1:
                    raise RuntimeError(
                        "Unexpected non-unary clade while collapsing the "
                        "known-tip subtree."
                    )

                child = active_children[0]
                if path_length is None or child.branch_length is None:
                    path_length = None
                else:
                    path_length += child.branch_length

            retained_children[parent].append((child, path_length))

    retained_preorder = [
        node
        for node in tree.find_clades(order="preorder")
        if node in retained
    ]

    # For an unknown node state x, its descendant subtree has minimized
    # cost a*x^2 - 2*b*x + c. These coefficients are accumulated in
    # postorder after analytically minimizing each child state.
    quadratic: dict[Clade, tuple[float, float, float]] = {}

    for node in reversed(retained_preorder):
        if node in tip_value:
            # Observed tips are fixed boundary values rather than quadratics.
            continue

        coefficient = 0.0
        linear = 0.0
        constant = 0.0

        for child, path_length in retained_children[node]:
            edge_weight = _length_weight(path_length, weighted=weighted)

            if child in tip_value:
                child_value = tip_value[child]
                coefficient += edge_weight
                linear += edge_weight * child_value
                constant += edge_weight * child_value * child_value
                continue

            child_coefficient, child_linear, child_constant = quadratic[child]
            if child_coefficient == 0.0:
                # No observed tip occurs below this child. Its entire clade
                # can take the parent's state at zero cost.
                continue

            denominator = child_coefficient + edge_weight
            coefficient += (
                edge_weight * child_coefficient / denominator
            )
            linear += edge_weight * child_linear / denominator
            constant += (
                child_constant
                - child_linear * child_linear / denominator
            )

        quadratic[node] = (coefficient, linear, constant)

    root_coefficient, root_linear, _ = quadratic[tree.root]
    if root_coefficient <= 0.0:
        # This is unreachable after the all-hidden check for a valid tree,
        # but retains castor's explicit success concept.
        return HSPResult(
            success=False,
            total_sum_of_squared_changes=float("nan"),
            states=np.array([], dtype=float),
        )

    state_by_clade: dict[Clade, float] = {
        tree.root: root_linear / root_coefficient
    }

    # Given a parent's globally optimal state, analytically recover each
    # child's globally optimal state. Observation-free clades inherit their
    # nearest reconstructed ancestor's state.
    for parent in retained_preorder:
        parent_state = state_by_clade[parent]

        for child, path_length in retained_children[parent]:
            if child in tip_value:
                state_by_clade[child] = tip_value[child]
                continue

            child_coefficient, child_linear, _ = quadratic[child]
            if child_coefficient == 0.0:
                state_by_clade[child] = parent_state
                continue

            edge_weight = _length_weight(path_length, weighted=weighted)
            state_by_clade[child] = (
                child_linear + edge_weight * parent_state
            ) / (child_coefficient + edge_weight)

    # castor forward-projects every state absent from the pruned subtree from
    # its nearest state-known ancestor. This includes hidden terminal clades
    # and unary nodes removed while collapsing the known-tip subtree.
    for parent in tree.find_clades(order="preorder"):
        if parent not in state_by_clade:
            raise RuntimeError("A tree clade was reached before its state.")
        for child in parent.clades:
            if child not in state_by_clade:
                state_by_clade[child] = state_by_clade[parent]

    internal_nodes = list(tree.get_nonterminals(order="preorder"))
    ordered_clades = tips + internal_nodes
    states = np.fromiter(
        (state_by_clade[clade] for clade in ordered_clades),
        dtype=float,
        count=len(ordered_clades),
    )

    total = math.fsum(
        _length_weight(path_length, weighted=weighted)
        * (state_by_clade[parent] - state_by_clade[child]) ** 2
        for parent in retained_preorder
        for child, path_length in retained_children[parent]
    )

    return HSPResult(
        success=True,
        total_sum_of_squared_changes=total,
        states=states,
    )
