"""Discrete Mk-model fitting used by the Bugphyzz ASR pipeline.

This module translates the path through ``phytools::fitMk`` used by
``export_bugphyzz.R``:

    fitMk(tree, x, model="ER", pi="equal")

Tip observations may be probability/likelihood vectors rather than only
hard state assignments. That is required by Bugphyzz, which initializes
unannotated tips with a uniform state vector.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import pandas as pd
from Bio.Phylo.BaseTree import Clade, Tree


@dataclass(frozen=True)
class FitMkOptimization:
    """Small optimizer summary analogous to ``fitMk$opt_results``."""

    convergence: int
    iterations: int
    evaluations: int
    message: str


@dataclass(frozen=True)
class FitMkResult:
    """Python counterpart of the fields returned by ``phytools::fitMk``."""

    log_likelihood: float
    rates: np.ndarray
    index_matrix: np.ndarray
    states: tuple[str, ...]
    pi: np.ndarray
    method: str
    root_prior: str
    optimization: FitMkOptimization
    tree: Tree
    data: pd.DataFrame

    @property
    def q_matrix(self) -> np.ndarray:
        """Return the fitted instantaneous transition-rate matrix."""
        return make_q(
            len(self.states),
            self.rates,
            self.index_matrix,
        )


@dataclass(frozen=True)
class AncrResult:
    """Marginal ancestral-state reconstruction returned by :func:`ancr`."""

    ace: pd.DataFrame
    log_likelihood: float
    reconstruction_type: tuple[str, str] = ("marginal", "global")


def make_q(
    state_count: int,
    rates: np.ndarray,
    index_matrix: np.ndarray,
) -> np.ndarray:
    """Translate phytools ``makeQ`` into NumPy."""
    indices = np.asarray(index_matrix, dtype=float)
    if indices.shape != (state_count, state_count):
        raise ValueError(
            "index_matrix must be square with one row per state."
        )

    rate_values = np.asarray(rates, dtype=float)
    q_matrix = np.zeros((state_count, state_count), dtype=float)

    for row in range(state_count):
        for column in range(state_count):
            index = indices[row, column]
            if np.isnan(index) or index == 0:
                continue
            integer_index = int(index)
            if integer_index < 1 or integer_index > len(rate_values):
                raise ValueError("index_matrix contains an invalid rate index.")
            q_matrix[row, column] = rate_values[integer_index - 1]

    np.fill_diagonal(q_matrix, 0.0)
    np.fill_diagonal(q_matrix, -q_matrix.sum(axis=1))
    return q_matrix


def _er_transition_matrix(
    rate: float,
    branch_length: float,
    state_count: int,
) -> np.ndarray:
    """Exponentiate an equal-rates Mk generator analytically."""
    decay = math.exp(-state_count * rate * branch_length)
    off_diagonal = (1.0 - decay) / state_count
    diagonal = off_diagonal + decay

    matrix = np.full(
        (state_count, state_count),
        off_diagonal,
        dtype=float,
    )
    np.fill_diagonal(matrix, diagonal)
    return matrix


def _er_propagate(
    likelihood: np.ndarray,
    rate: float,
    branch_length: float,
    state_count: int,
) -> np.ndarray:
    """Calculate ``expm(Q*t) @ likelihood`` without forming a matrix."""
    decay = math.exp(-state_count * rate * branch_length)
    off_diagonal = (1.0 - decay) / state_count
    return decay * likelihood + off_diagonal * likelihood.sum()


def _prepare_tip_likelihoods(
    tree: Tree,
    x: pd.DataFrame,
) -> tuple[list[Clade], pd.DataFrame, tuple[str, ...]]:
    """Validate and reorder a phytools-style tip likelihood matrix."""
    if not isinstance(x, pd.DataFrame):
        raise TypeError(
            "x must be a pandas DataFrame indexed by tree tip label."
        )
    if x.empty or x.shape[1] < 2:
        raise ValueError("x must contain at least two discrete states.")
    if not x.index.is_unique:
        raise ValueError("x contains duplicate tip labels.")

    tips = list(tree.get_terminals(order="preorder"))
    labels: list[str] = []
    for tip in tips:
        if tip.name is None:
            raise ValueError("The tree contains an unnamed terminal tip.")
        labels.append(str(tip.name))

    missing = [label for label in labels if label not in x.index]
    extra = [str(label) for label in x.index if str(label) not in set(labels)]
    if missing or extra:
        raise ValueError(
            "x and the tree must contain the same tip labels. "
            f"Missing: {missing[:10]}; extra: {extra[:10]}."
        )

    ordered = x.loc[labels].copy()
    try:
        values = ordered.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("x must contain numeric likelihood values.") from exc

    if not np.isfinite(values).all():
        raise ValueError("x contains missing or non-finite likelihoods.")
    if (values < 0).any():
        raise ValueError("x contains negative likelihoods.")
    if (values.sum(axis=1) <= 0).any():
        raise ValueError("Every tip must allow at least one state.")

    ordered.iloc[:, :] = values
    states = tuple(str(column) for column in ordered.columns)
    return tips, ordered, states


def _er_negative_log_likelihood(
    postorder_nodes: list[Clade],
    child_edges: dict[Clade, list[tuple[Clade, float]]],
    root: Clade,
    tip_likelihoods: dict[Clade, np.ndarray],
    *,
    rate: float,
    state_count: int,
    root_prior: np.ndarray,
) -> float:
    """Evaluate phytools' scaled postorder pruning likelihood."""
    if not math.isfinite(rate) or rate < 0:
        return float("inf")

    likelihoods = dict(tip_likelihoods)
    negative_log_likelihood = 0.0

    for node in postorder_nodes:
        combined = np.ones(state_count, dtype=float)

        for child, branch_length in child_edges[node]:
            combined *= _er_propagate(
                likelihoods[child],
                rate,
                branch_length,
                state_count,
            )

        if node is root:
            combined *= root_prior

        scale = float(combined.sum())
        if not math.isfinite(scale) or scale <= 0:
            return float("inf")

        negative_log_likelihood -= math.log(scale)
        likelihoods[node] = combined / scale

    return negative_log_likelihood


def _bounded_minimize(
    objective: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
    max_iterations: int = 256,
) -> tuple[float, float, FitMkOptimization]:
    """Minimize a scalar function using bounded golden-section search."""
    if lower < 0 or not math.isfinite(lower):
        raise ValueError("The lower rate bound must be finite and nonnegative.")
    if not math.isfinite(upper) or upper <= lower:
        raise ValueError("The upper rate bound must exceed the lower bound.")

    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = lower
    right = upper
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = objective(x1)
    f2 = objective(x2)
    evaluations = 2
    iterations = 0

    while iterations < max_iterations:
        tolerance = absolute_tolerance + relative_tolerance * max(
            abs(left), abs(right)
        )
        if right - left <= tolerance:
            break

        if f1 <= f2:
            right = x2
            x2 = x1
            f2 = f1
            x1 = right - ratio * (right - left)
            f1 = objective(x1)
        else:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + ratio * (right - left)
            f2 = objective(x2)

        evaluations += 1
        iterations += 1

    candidates = [
        (left, objective(left)),
        (x1, f1),
        (x2, f2),
        (right, objective(right)),
    ]
    evaluations += 2
    optimum, value = min(candidates, key=lambda item: item[1])

    optimization = FitMkOptimization(
        convergence=0 if iterations < max_iterations else 1,
        iterations=iterations,
        evaluations=evaluations,
        message=(
            "bounded scalar convergence"
            if iterations < max_iterations
            else "iteration limit reached"
        ),
    )
    return optimum, value, optimization


def _maximum_root_distance(tree: Tree) -> float:
    """Return ``max(nodeHeights(tree))`` for a rooted Biopython tree."""
    maximum = 0.0
    stack: list[tuple[Clade, float]] = [(tree.root, 0.0)]

    while stack:
        node, height = stack.pop()
        maximum = max(maximum, height)
        for child in node.clades:
            length = 1.0 if child.branch_length is None else float(child.branch_length)
            if not math.isfinite(length) or length < 0:
                raise ValueError("The tree contains an invalid branch length.")
            stack.append((child, height + length))

    return maximum


def fit_mk(
    tree: Tree,
    x: pd.DataFrame,
    model: str = "ER",
    *,
    pi: str | np.ndarray = "equal",
    q_init: float | None = None,
    min_q: float = 0.0,
    max_q: float | None = None,
    opt_method: str = "bounded",
) -> FitMkResult:
    """Fit the equal-rates Mk model used by Bugphyzz.

    This translates the ``model="ER"``, ``pi="equal"`` path through
    ``phytools::fitMk``. ``SYM``, ``ARD``, FitzJohn, stationary, and fitted
    root priors are deliberately not accepted until corresponding frozen
    parity fixtures are available.
    """
    if model != "ER":
        raise NotImplementedError(
            "Only phytools fitMk model='ER' is currently implemented."
        )

    tips, ordered, states = _prepare_tip_likelihoods(tree, x)
    state_count = len(states)

    if isinstance(pi, str):
        if pi != "equal":
            raise NotImplementedError(
                "Only pi='equal' or an explicit numeric prior is implemented."
            )
        root_prior = np.full(state_count, 1.0 / state_count)
        root_prior_name = "flat"
    else:
        root_prior = np.asarray(pi, dtype=float)
        if root_prior.shape != (state_count,):
            raise ValueError("pi must contain one probability per state.")
        if (
            not np.isfinite(root_prior).all()
            or (root_prior < 0).any()
            or root_prior.sum() <= 0
        ):
            raise ValueError("pi must contain valid nonnegative probabilities.")
        root_prior = root_prior / root_prior.sum()
        root_prior_name = "given"

    values = ordered.to_numpy(dtype=float)
    tip_likelihoods = {
        tip: values[index]
        for index, tip in enumerate(tips)
    }
    postorder_nodes = list(tree.get_nonterminals(order="postorder"))
    child_edges: dict[Clade, list[tuple[Clade, float]]] = {}
    for node in postorder_nodes:
        edges: list[tuple[Clade, float]] = []
        for child in node.clades:
            length = (
                1.0
                if child.branch_length is None
                else float(child.branch_length)
            )
            if not math.isfinite(length) or length < 0:
                raise ValueError("The tree contains an invalid branch length.")
            edges.append((child, length))
        child_edges[node] = edges

    total_branch_length = math.fsum(
        1.0 if child.branch_length is None else float(child.branch_length)
        for node in tree.find_clades(order="preorder")
        for child in node.clades
    )
    if total_branch_length <= 0:
        raise ValueError("The tree must have positive total branch length.")

    if q_init is None:
        q_init = state_count / total_branch_length
    if not math.isfinite(q_init) or q_init < 0:
        raise ValueError("q_init must be finite and nonnegative.")

    if max_q is None:
        max_q = _maximum_root_distance(tree) * 100.0
    if max_q <= 0:
        raise ValueError("max_q must be positive.")

    def objective(rate: float) -> float:
        return _er_negative_log_likelihood(
            postorder_nodes,
            child_edges,
            tree.root,
            tip_likelihoods,
            rate=rate,
            state_count=state_count,
            root_prior=root_prior,
        )

    if opt_method == "bounded":
        fitted_rate, negative_log_likelihood, optimization = _bounded_minimize(
            objective,
            min_q,
            max_q,
        )
        method = "bounded_scalar"
    elif opt_method == "none":
        fitted_rate = q_init
        negative_log_likelihood = objective(fitted_rate)
        optimization = FitMkOptimization(
            convergence=0,
            iterations=0,
            evaluations=1,
            message="rate supplied without optimization",
        )
        method = "none"
    else:
        raise ValueError("opt_method must be 'bounded' or 'none'.")

    index_matrix = np.ones((state_count, state_count), dtype=float)
    np.fill_diagonal(index_matrix, np.nan)

    return FitMkResult(
        log_likelihood=-negative_log_likelihood,
        rates=np.array([fitted_rate], dtype=float),
        index_matrix=index_matrix,
        states=states,
        pi=root_prior,
        method=method,
        root_prior=root_prior_name,
        optimization=optimization,
        tree=tree,
        data=ordered,
    )


def ancr(
    fit: FitMkResult,
    *,
    tips: bool = False,
    reconstruction_type: str = "marginal",
) -> AncrResult:
    """Reconstruct global marginal Mk state probabilities.

    This translates the relevant path through
    ``phytools::ancr.fitMk``:

    ``ancr(fit, type="marginal", tips=TRUE, lik.func="pruning")``.

    The implementation first performs phytools' scaled postorder pruning,
    then applies its root-to-tip ``marginal_asr`` update. Only ER
    :class:`FitMkResult` objects are currently produced by :func:`fit_mk`.
    """
    if reconstruction_type != "marginal":
        raise NotImplementedError(
            "Only global marginal reconstruction is implemented."
        )
    if len(fit.rates) != 1:
        raise NotImplementedError(
            "ancr currently supports equal-rates FitMkResult objects."
        )

    tree = fit.tree
    state_count = len(fit.states)
    rate = float(fit.rates[0])
    tip_clades = list(tree.get_terminals(order="preorder"))
    tip_values = fit.data.to_numpy(dtype=float)

    conditional: dict[Clade, np.ndarray] = {
        tip: tip_values[index].copy()
        for index, tip in enumerate(tip_clades)
    }
    log_likelihood = 0.0

    # phytools::pruning(..., return="conditional")
    for node in tree.get_nonterminals(order="postorder"):
        combined = np.ones(state_count, dtype=float)

        for child in node.clades:
            branch_length = (
                1.0
                if child.branch_length is None
                else float(child.branch_length)
            )
            combined *= _er_propagate(
                conditional[child],
                rate,
                branch_length,
                state_count,
            )

        if node is tree.root:
            combined *= fit.pi

        scale = float(combined.sum())
        if not math.isfinite(scale) or scale <= 0:
            raise FloatingPointError(
                "Mk pruning produced a nonpositive likelihood scale."
            )

        log_likelihood += math.log(scale)
        conditional[node] = combined / scale

    # phytools::marginal_asr traverses the postorder parents in reverse,
    # which is equivalent to a root-first traversal of the edges.
    for parent in tree.find_clades(order="preorder"):
        parent_likelihood = conditional[parent]

        for child in parent.clades:
            branch_length = (
                1.0
                if child.branch_length is None
                else float(child.branch_length)
            )
            child_likelihood = conditional[child]
            denominator = _er_propagate(
                child_likelihood,
                rate,
                branch_length,
                state_count,
            )
            parent_without_child = np.divide(
                parent_likelihood,
                denominator,
                out=np.zeros_like(parent_likelihood),
                where=denominator != 0,
            )
            conditional[child] = (
                _er_propagate(
                    parent_without_child,
                    rate,
                    branch_length,
                    state_count,
                )
                * child_likelihood
            )

    internal_clades = list(tree.get_nonterminals(order="preorder"))
    output_clades = (
        tip_clades + internal_clades
        if tips
        else internal_clades
    )
    labels: list[str] = []
    probabilities = np.empty(
        (len(output_clades), state_count),
        dtype=float,
    )

    for index, clade in enumerate(output_clades):
        if clade.name is None:
            raise ValueError("The tree contains an unnamed output clade.")

        labels.append(str(clade.name))
        row = conditional[clade]
        total = float(row.sum())
        if not math.isfinite(total) or total <= 0:
            raise FloatingPointError(
                "Marginal reconstruction produced an invalid probability row."
            )
        probabilities[index] = row / total

    ace = pd.DataFrame(
        probabilities,
        index=pd.Index(labels, name="label"),
        columns=list(fit.states),
    )
    return AncrResult(
        ace=ace,
        log_likelihood=log_likelihood,
    )
