"""
N-gram extraction utilities for AGraph expressions.

Extracts operator n-grams from an AGraph's tree structure for the
Katz back-off prior of Bartlett et al. (GECCO 2023, Sec 2.4).

The meaning of ``n`` follows the **paper text** (not the reference
code, which is off-by-one):

- ``n`` is the **n-gram order** (phrase length):
  - ``n=1``: unigrams — no parent conditioning
  - ``n=2``: bigrams — condition on the immediate parent
  - ``n=3``: trigrams — condition on parent + grandparent

- Phrases have **variable length up to** ``n``. Nodes near the top of
  the tree, with fewer than ``n-1`` ancestors, naturally produce
  shorter phrases. **No ROOT/sentinel padding** is used.
- For each non-terminal node the authors emit *one* phrase capturing
  its sibling structure:

    - **unary** node ``op``: left phrase ``(ancestors..., child)``.
    - **binary** node ``op``: left phrase
      ``(ancestors..., left_sibling)`` and right phrase
      ``(ancestors[1:]..., left_sibling, right_sibling)``.

  Note the **drop of the oldest ancestor** for the right phrase.
- A leading singleton ``(root_op,)`` is added to the left list to
  represent ``P(root)`` (the unigram start of the expression tree).

Notes on paper vs reference code
--------------------------------
The paper text says "n=1 means we do not consider the effect of any
parents, n=2 the probabilities are conditioned on parent nodes only,
and n=3 we condition on both parent and grandparent nodes." However,
the reference code at https://github.com/DeaglanBartlett/katz has an
off-by-one: its ``n=2`` produces trigrams (length-3 phrases). This
module follows the **paper's** convention so that Figure 1 labels
match the paper.
"""

from typing import List, Optional, Tuple

from bingo.expressions.agraph import AGraphExpression
from bingo.expressions.agraph.evolvable import EvolvableExpression
from bingo.expressions.agraph.pyagraph import IS_ARITY_2_ARRAY, IS_TERMINAL_ARRAY


# Sentinel kept for backward compatibility with code/JSON that may still
# reference it. The current implementation does **not** pad phrases,
# so this token should not appear in extracted phrases or vocabularies.
ROOT: int = -1


def _unwrap(agraph):
    """Return the underlying AGraphExpression, stripping evolvable wrappers."""
    if isinstance(agraph, EvolvableExpression):
        return agraph.expression
    return agraph


def extract_phrases(
    agraph: AGraphExpression, n: int
) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    """Extract Katz back-off phrases from an AGraph expression tree.

    Walks the AGraph's ``command_array`` depth-first. For every
    non-terminal node ``op`` with parent context
    ``ancestors = (a_1, ..., a_k)`` (clipped on the *left* to the most
    recent ``n - 1`` operators, no padding) emits:

    - **Unary**: a single left/only phrase
      ``(ancestors..., child_op)``.
    - **Binary**: a left/only phrase
      ``(ancestors..., left_sibling_op)`` *and* a right phrase
      ``(ancestors[1:]..., left_sibling_op, right_sibling_op)``.

    A leading singleton ``(root_op,)`` is added to the left list to
    cover the unconditioned probability of the tree root.

    Parameters
    ----------
    agraph : AGraphExpression
        Expression to extract phrases from. Evolvable wrappers are
        unwrapped automatically.
    n : int
        The n-gram order (phrase length). ``n=1`` means unigrams (no
        parent conditioning), ``n=2`` conditions on the parent only,
        ``n=3`` conditions on parent + grandparent, etc. Must be
        ``>= 1``.

    Returns
    -------
    left_phrases : list of tuple of int
        Phrases of length 1 .. ``n`` for the left/only model.
    right_phrases : list of tuple of int
        Phrases of length 2 .. ``n`` for the right-given-left
        model. Empty when the tree has no binary operators.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    agraph = _unwrap(agraph)
    cmd = agraph.command_array
    if cmd.shape[0] == 0:
        return [], []

    root_idx = cmd.shape[0] - 1
    root_op = int(cmd[root_idx, 0])

    # Paper's authors prepend a singleton (root_op,) representing P(root).
    left_phrases: List[Tuple[int, ...]] = [(root_op,)]
    right_phrases: List[Tuple[int, ...]] = []

    # DFS with per-node ancestor context (oldest -> newest).
    # Paper's n is the n-gram order (phrase length), so we keep n-1 ancestors.
    n_ancestors = n - 1
    stack: List[Tuple[int, Tuple[int, ...]]] = [(root_idx, ())]
    while stack:
        idx, ancestors = stack.pop()
        row = cmd[idx]
        op = int(row[0])

        if IS_TERMINAL_ARRAY[op]:
            continue

        # Keep at most n-1 ancestors; phrase will have length up to n.
        if n_ancestors > 0:
            child_chain = (ancestors + (op,))[-n_ancestors:]
        else:
            child_chain = ()

        if IS_ARITY_2_ARRAY[op]:
            left_idx = int(row[1])
            right_idx = int(row[2])
            left_op = int(cmd[left_idx, 0])
            right_op = int(cmd[right_idx, 0])

            # Left phrase: full ancestor chain + left sibling.
            left_phrases.append(child_chain + (left_op,))
            # Right phrase: oldest ancestor dropped + (left, right).
            right_phrases.append(child_chain[1:] + (left_op, right_op))

            stack.append((left_idx, child_chain))
            stack.append((right_idx, child_chain))
        else:
            only_idx = int(row[1])
            only_op = int(cmd[only_idx, 0])
            left_phrases.append(child_chain + (only_op,))
            stack.append((only_idx, child_chain))

    return left_phrases, right_phrases


def collect_phrases(
    agraphs: List[AGraphExpression], n: int
) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    """Extract phrases from a collection of AGraph expressions."""
    all_left: List[Tuple[int, ...]] = []
    all_right: List[Tuple[int, ...]] = []
    for ag in agraphs:
        lefts, rights = extract_phrases(ag, n)
        all_left.extend(lefts)
        all_right.extend(rights)
    return all_left, all_right


def vocabulary(
    phrases: List[Tuple[int, ...]],
    extra: Optional[List[int]] = None,
) -> List[int]:
    """Return a sorted list of unique operator tokens appearing in phrases."""
    vocab = set()
    for phrase in phrases:
        for tok in phrase:
            if tok != ROOT:
                vocab.add(tok)
    if extra is not None:
        for tok in extra:
            if tok != ROOT:
                vocab.add(tok)
    return sorted(vocab)
