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

Notices:
Copyright 2025 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.
 
The NASA Software “PySIPS” (LAR-20644-1) calls the following third-party software, which is subject to the terms and conditions of its licensor, as applicable at the time of licensing.  The third-party software is not bundled or included with this software but may be available from the licensor.  License hyperlinks are provided here for information purposes only.
 
NumPy  
https://numpy.org/devdocs/license.html  
Copyright (c) 2005-2025, NumPy Developers.  
All rights reserved.
 
h5py  
https://github.com/h5py/h5py/blob/master/LICENSE  
Copyright (c) 2008 Andrew Collette and contributors  
All rights reserved.
 
tqdm  
https://github.com/tqdm/tqdm/blob/master/LICENCE  
Copyright (c) 2013 noamraph
 
SciPy  
https://github.com/scipy/scipy/blob/main/LICENSE.txt  
Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.  
All rights reserved.
 
Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."
 
Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS, AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.


## Acknowledgements

This work was developed at NASA Langley Research Center.