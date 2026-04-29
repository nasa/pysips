"""Convert the Feynman Symbolic Regression Database CSVs into the
`wikipedia_named_equations.txt`-style corpus format used by
:func:`pysips.priors.data.load_corpus.load_corpus`.

The input CSVs (``FeynmanEquations.csv`` and ``BonusEquations.csv``) come
from the FSReD dataset distributed with AI Feynman / Udrescu & Tegmark
(2020). Each row has a ``Formula`` column (written in standard math
notation referring to variables like ``theta``, ``m``, ``c``, ...) and
a list of variable names in adjacent columns.

This script:

1. Parses each formula via :func:`sympy.sympify`, substituting every
   variable name with ``Symbol('X_i')`` (ordered as the CSV lists them).
2. Replaces :data:`sympy.pi` with its numeric ``Float`` value so that
   the resulting expression can be parsed by ``AGraphExpression`` —
   bingo has no first-class ``pi`` operator.
3. Writes one :func:`sympy.srepr` string per line to the output file.

Run from the pysips repo root::

    python -m pysips.priors.data.convert_feynman_csv
"""

import csv
from pathlib import Path

from sympy import Float, Symbol, pi as sym_pi, srepr, sympify


DATA_DIR = Path(__file__).parent
FEYNMAN_CSV = DATA_DIR / "FeynmanEquations.csv"
BONUS_CSV = DATA_DIR / "BonusEquations.csv"
BENCHMARK_CSV = DATA_DIR / "BenchmarkFeynmanEquations.csv"
FEYNMAN_OUT = DATA_DIR / "feynman_equations.txt"
BONUS_OUT = DATA_DIR / "bonus_equations.txt"
BENCHMARK_OUT = DATA_DIR / "benchmark_feynman_equations.txt"


# Column layout differs between the two CSVs. The two integers are
# (formula_column_index, first_variable_name_column_index).
FEYNMAN_LAYOUT = (3, 5)
BONUS_LAYOUT = (5, 7)
BENCHMARK_LAYOUT = (3, 5)


def _iter_equations(csv_path, formula_col, var_start_col):
    """Yield ``(formula_string, [variable_names])`` for each non-empty row."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if not row or not row[formula_col].strip():
                continue
            formula = row[formula_col].strip()
            var_names = []
            i = var_start_col
            while i < len(row) and row[i].strip():
                var_names.append(row[i].strip())
                i += 3  # columns are (name, low, high) triplets
            yield formula, var_names


def _convert(formula, var_names):
    """Convert a single formula string to its srepr form."""
    locals_ = {name: Symbol(f"X_{i}") for i, name in enumerate(var_names)}
    expr = sympify(formula, locals=locals_)
    expr = expr.subs(sym_pi, Float(float(sym_pi)))
    return srepr(expr)


def _write_corpus(lines, out_path):
    """Write lines to corpus file."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def build_feynman_corpus():
    """Write the Feynman Lectures corpus (100 equations)."""
    lines = [_convert(f, v) for f, v in _iter_equations(FEYNMAN_CSV, *FEYNMAN_LAYOUT)]
    return _write_corpus(lines, FEYNMAN_OUT)


def build_bonus_corpus():
    """Write the bonus corpus (20 equations from Goldstein, Jackson, etc.)."""
    lines = [_convert(f, v) for f, v in _iter_equations(BONUS_CSV, *BONUS_LAYOUT)]
    return _write_corpus(lines, BONUS_OUT)


def build_benchmark_corpus():
    """Write the benchmark Feynman corpus."""
    lines = [
        _convert(f, v) for f, v in _iter_equations(BENCHMARK_CSV, *BENCHMARK_LAYOUT)
    ]
    return _write_corpus(lines, BENCHMARK_OUT)


if __name__ == "__main__":
    n_feyn = build_feynman_corpus()
    n_bonus = build_bonus_corpus()
    n_bench = build_benchmark_corpus()
    print(f"Wrote {n_feyn} equations to {FEYNMAN_OUT}")
    print(f"Wrote {n_bonus} equations to {BONUS_OUT}")
    print(f"Wrote {n_bench} equations to {BENCHMARK_OUT}")
