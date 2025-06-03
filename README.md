# PySIPS: Python package for Symbolic Inference via Posterior Sampling

PySIPS is an open-source implementation of Bayesian symbolic regression via posterior sampling as described in the paper "Bayesian Symbolic Regression via Posterior Sampling" by G. F. Bomarito and P. E. Leser from NASA Langley Research Center.

## Purpose

PySIPS provides a robust framework for discovering interpretable symbolic expressions from data, with a particular focus on handling noisy datasets. Unlike traditional symbolic regression approaches, PySIPS uses a Bayesian framework with Sequential Monte Carlo (SMC) sampling to:

1. Enhance robustness to noise
2. Provide built-in uncertainty quantification
3. Discover parsimonious expressions with improved generalization
4. Reduce overfitting in symbolic regression tasks

## Algorithm Overview

PySIPS implements a Sequential Monte Carlo (SMC) framework for Bayesian symbolic regression that:

- Approximates the posterior distribution over symbolic expressions
- Uses probabilistic selection and adaptive annealing to explore the search space efficiently
- Employs normalized marginal likelihood for model evaluation
- Combines mutation and crossover operations as proposal mechanisms
- Provides model selection criteria based on maximum normalized marginal likelihood or posterior mode

## Installation

(Coming Soon!)

```bash
pip install pysips
```

## Example Usage

```python
import numpy as np
from pysips import PysipsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Generate synthetic data (y = x^2 + noise)
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = X[:, 0]**2 + np.random.normal(0, 0.1, size=X.shape[0])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the regressor
regressor = PysipsRegressor(
    operators=['+', '-', '*', '^2'],
    max_complexity=12,
    num_particles=100,
    num_mcmc_samples=10,
    random_state=42
)

regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Get the discovered expression
expression = regressor.get_expression()
print(f"Discovered expression: {expression}")
print(f"R² score: {r2_score(y_test, y_pred):.4f}")

# Get model posterior and their likelihoods
models, likelihoods = regressor.get_models()
```

### Example Output

```
Discovered expression: x_0^2
R² score: 0.9987
Number of unique models sampled: 32
```

## Advanced Features

- Control over operators and expression complexity
- Multiple model selection strategies
- Access to the full posterior distribution over expressions
- Compatible with scikit-learn's API for easy integration into ML pipelines
- Uncertainty quantification for symbolic regression results

## Citation

If you use PySIPS, please cite the following paper:

```bibtex
@article{bomarito2024bayesian,
  title={Bayesian Symbolic Regression via Posterior Sampling},
  author={Bomarito, Geoffrey F. and Leser, Patrick E.},
  journal={Philosophical Transactions of the Royal Society A},
  year={2025},
  publisher={Royal Society}
}
```

## License

[Insert your chosen license here]


## Acknowledgements

This work was developed at NASA Langley Research Center.