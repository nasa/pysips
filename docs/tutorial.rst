Tutorial
========

This tutorial demonstrates key features of PySIPS through practical examples and common use cases.

Table of Contents
-----------------

1. :ref:`basic-symbolic-regression`
2. :ref:`multivariate-regression`
3. :ref:`custom-operators`
4. :ref:`model-selection-strategies`
5. :ref:`checkpointing`
6. :ref:`hyperparameter-tuning`
7. :ref:`posterior-analysis`

.. _basic-symbolic-regression:

1. Basic Symbolic Regression
-----------------------------

Let's start with a simple univariate symbolic regression problem:

.. code-block:: python

   import numpy as np
   from pysips import PysipsRegressor
   import matplotlib.pyplot as plt

   # Generate data: y = sin(x) + noise
   np.random.seed(42)
   X = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
   y = np.sin(X[:, 0]) + np.random.normal(0, 0.1, size=X.shape[0])

   # Create and fit the regressor
   regressor = PysipsRegressor(
       operators=['+', '-', '*', '/'],
       max_complexity=15,
       num_particles=100,
       num_mcmc_samples=10,
       max_time=120,
       random_state=42
   )
   
   regressor.fit(X, y)
   
   # Get results
   expression = regressor.get_expression()
   y_pred = regressor.predict(X)
   
   print(f"Discovered expression: {expression}")
   
   # Visualize results
   plt.figure(figsize=(10, 6))
   plt.scatter(X, y, alpha=0.5, label='Data')
   plt.plot(X, y_pred, 'r-', linewidth=2, label='Predicted')
   plt.xlabel('x')
   plt.ylabel('y')
   plt.legend()
   plt.title(f'Discovered: {expression}')
   plt.show()

.. _multivariate-regression:

2. Multivariate Regression
---------------------------

PySIPS handles multiple input variables naturally:

.. code-block:: python

   import numpy as np
   from pysips import PysipsRegressor
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import r2_score, mean_squared_error

   # Generate data: y = x0^2 + 2*x1 - 3*x2 + noise
   np.random.seed(42)
   n_samples = 200
   X = np.random.randn(n_samples, 3)
   y = X[:, 0]**2 + 2*X[:, 1] - 3*X[:, 2] + np.random.normal(0, 0.1, n_samples)

   # Split the data
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # Create regressor with power operator
   regressor = PysipsRegressor(
       operators=['+', '-', '*', 'pow'],
       max_complexity=20,
       num_particles=150,
       num_mcmc_samples=10,
       max_time=180,
       random_state=42
   )

   # Fit and evaluate
   regressor.fit(X_train, y_train)
   y_pred = regressor.predict(X_test)
   
   expression = regressor.get_expression()
   r2 = r2_score(y_test, y_pred)
   mse = mean_squared_error(y_test, y_pred)
   
   print(f"Discovered expression: {expression}")
   print(f"R² score: {r2:.4f}")
   print(f"MSE: {mse:.4f}")

.. _custom-operators:

3. Customizing Available Operators
-----------------------------------

You can control which mathematical operators are available:

.. code-block:: python

   from pysips import PysipsRegressor

   # Example 1: Simple arithmetic only
   regressor_simple = PysipsRegressor(
       operators=['+', '-', '*'],  # No division
       max_complexity=10
   )

   # Example 2: Include division and power
   regressor_extended = PysipsRegressor(
       operators=['+', '-', '*', '/', 'pow'],
       max_complexity=15
   )

   # Example 3: Include trigonometric functions
   regressor_trig = PysipsRegressor(
       operators=['+', '-', '*', 'sin', 'cos'],
       max_complexity=15
   )

   # Example 4: Include exponential and logarithm
   regressor_exp = PysipsRegressor(
       operators=['+', '-', '*', 'exp', 'log'],
       max_complexity=15
   )

**Available operators:**

- Arithmetic: ``'+'``, ``'-'``, ``'*'``, ``'/'``
- Power: ``'pow'``, ``'sqrt'``
- Trigonometric: ``'sin'``, ``'cos'``, ``'tan'``
- Exponential/Logarithmic: ``'exp'``, ``'log'``
- Other: ``'abs'``

.. _model-selection-strategies:

4. Model Selection Strategies
------------------------------

PySIPS offers two strategies for selecting the final model:

.. code-block:: python

   import numpy as np
   from pysips import PysipsRegressor

   X = np.random.randn(100, 2)
   y = X[:, 0]**2 + X[:, 1] + np.random.normal(0, 0.1, 100)

   # Strategy 1: Mode selection (most frequently sampled)
   regressor_mode = PysipsRegressor(
       operators=['+', '*', 'pow'],
       model_selection='mode',  # Default
       num_particles=100,
       random_state=42
   )
   regressor_mode.fit(X, y)
   print(f"Mode selection: {regressor_mode.get_expression()}")

   # Strategy 2: Maximum likelihood selection
   regressor_maxlik = PysipsRegressor(
       operators=['+', '*', 'pow'],
       model_selection='max_likelihood',
       num_particles=100,
       random_state=42
   )
   regressor_maxlik.fit(X, y)
   print(f"Max likelihood selection: {regressor_maxlik.get_expression()}")

**When to use each:**

- ``'mode'``: More robust, favors expressions that appear frequently in the posterior
- ``'max_likelihood'``: Favors the single best-scoring expression, may overfit

.. _checkpointing:

5. Checkpointing for Long Runs
-------------------------------

For long-running experiments, use checkpointing to save progress:

.. code-block:: python

   import numpy as np
   from pysips import PysipsRegressor

   X = np.random.randn(500, 3)
   y = np.sum(X**2, axis=1) + np.random.normal(0, 0.1, 500)

   # Specify a checkpoint file
   regressor = PysipsRegressor(
       operators=['+', '-', '*', 'pow'],
       max_complexity=25,
       num_particles=200,
       checkpoint_file='my_experiment.checkpoint',  # Save/load from this file
       max_time=600,  # 10 minutes
       random_state=42
   )

   # First run: starts from scratch and saves progress
   regressor.fit(X, y)
   
   # If interrupted, subsequent runs will resume from checkpoint
   # regressor.fit(X, y)  # Resumes from saved state

**Checkpoint behavior:**

- If checkpoint file doesn't exist: starts fresh, creates checkpoint
- If checkpoint file exists: resumes from saved state
- Progress is saved periodically during fitting
- Useful for long experiments that might be interrupted

.. _hyperparameter-tuning:

6. Hyperparameter Tuning
-------------------------

PySIPS is compatible with scikit-learn's hyperparameter tuning tools:

.. code-block:: python

   import numpy as np
   from pysips import PysipsRegressor
   from sklearn.model_selection import GridSearchCV, cross_val_score
   from sklearn.datasets import make_friedman1

   # Generate data
   X, y = make_friedman1(n_samples=200, n_features=5, noise=0.1, random_state=42)

   # Define parameter grid
   param_grid = {
       'max_complexity': [10, 15, 20],
       'num_particles': [50, 100, 150],
   }

   # Create base regressor (disable progress bar for cleaner output)
   base_regressor = PysipsRegressor(
       operators=['+', '-', '*'],
       num_mcmc_samples=10,
       max_time=60,
       show_progress_bar=False,  # Important for grid search
       random_state=42
   )

   # Perform grid search
   grid_search = GridSearchCV(
       base_regressor,
       param_grid,
       cv=3,
       scoring='r2',
       n_jobs=1  # PySIPS doesn't support parallelization at this level
   )

   grid_search.fit(X, y)

   print(f"Best parameters: {grid_search.best_params_}")
   print(f"Best score: {grid_search.best_score_:.4f}")
   
   # Use best estimator
   best_expression = grid_search.best_estimator_.get_expression()
   print(f"Best expression: {best_expression}")

.. _posterior-analysis:

7. Analyzing the Posterior Distribution
----------------------------------------

Access the full posterior distribution for uncertainty quantification:

.. code-block:: python

   import numpy as np
   from pysips import PysipsRegressor
   from collections import Counter

   # Generate data
   np.random.seed(42)
   X = np.random.randn(100, 2)
   y = X[:, 0]**2 + 2*X[:, 1] + np.random.normal(0, 0.1, 100)

   # Fit regressor
   regressor = PysipsRegressor(
       operators=['+', '-', '*', 'pow'],
       max_complexity=15,
       num_particles=200,
       num_mcmc_samples=10,
       max_time=120,
       random_state=42
   )
   regressor.fit(X, y)

   # Get all sampled models and their likelihoods
   models, likelihoods = regressor.get_models()

   print(f"Total unique models sampled: {len(models)}")
   print(f"\nTop 5 models by likelihood:")
   
   # Sort by likelihood
   sorted_indices = np.argsort(likelihoods)[::-1]
   
   for i in range(min(5, len(models))):
       idx = sorted_indices[i]
       print(f"{i+1}. {models[idx]} (likelihood: {likelihoods[idx]:.4f})")

   # Count frequency of each model in the posterior
   print(f"\nSelected model: {regressor.get_expression()}")
   print(f"(This is the most frequently sampled model)")

Advanced Topics
---------------

**Mutation and Crossover Control:**

Fine-tune the proposal mechanism probabilities:

.. code-block:: python

   regressor = PysipsRegressor(
       operators=['+', '-', '*'],
       # Mutation probabilities
       command_probability=0.3,    # Probability of changing an operator
       node_probability=0.3,        # Probability of replacing a node
       parameter_probability=0.2,   # Probability of changing edges in an expression graph
       prune_probability=0.1,       # Probability of simplifying
       fork_probability=0.1,        # Probability of expanding
       # Crossover settings
       crossover_pool_size=20,      # Size of gene pool for crossover
       random_state=42
   )

**Expression Complexity Control:**

.. code-block:: python

   regressor = PysipsRegressor(
       operators=['+', '-', '*'],
       max_complexity=10,              # Maximum expression size
       terminal_probability=0.5,       # Probability of leaf nodes
       constant_probability=0.3,       # Probability of constants vs variables
       random_state=42
   )

**SMC Sampling Control:**

.. code-block:: python

   regressor = PysipsRegressor(
       operators=['+', '-', '*'],
       num_particles=100,          # Population size
       num_mcmc_samples=10,        # MCMC steps per SMC iteration
       target_ess=0.8,             # Target effective sample size (0-1)
       random_state=42
   )

Best Practices
--------------

1. **Start simple**: Begin with basic operators and small complexity limits
2. **Use checkpoints**: For experiments taking >5 minutes, always use checkpointing
3. **Reproducibility**: Always set ``random_state`` for reproducible results
4. **Progress monitoring**: Use ``show_progress_bar=True`` (default) for interactive use
5. **Hyperparameter tuning**: Disable progress bar when using GridSearchCV
6. **Model validation**: Always validate on held-out test data
7. **Posterior inspection**: Check the full posterior distribution, not just the best model

Troubleshooting
---------------

**Issue: Fitting is too slow**

- Reduce ``num_particles`` (e.g., 50-100)
- Reduce ``max_complexity`` (e.g., 10-15)
- Set ``max_time`` to limit runtime
- Simplify the operator set

**Issue: Poor model quality**

- Increase ``num_particles`` (e.g., 200-500)
- Increase ``num_mcmc_samples`` (e.g., 20-50)
- Increase ``max_complexity`` if needed
- Add more relevant operators
- Collect more/better training data

**Issue: Overfitting**

- Use cross-validation for model selection
- Reduce ``max_complexity``
- Use ``model_selection='mode'`` instead of ``'max_likelihood'``

**Issue: Checkpoint file corrupted**

- Delete the checkpoint file and start fresh
- Ensure sufficient disk space

Further Resources
-----------------

- `GitHub Repository <https://github.com/nasa/pysips>`_
- `API Reference <api/modules.html>`_
- `Paper: Bayesian Symbolic Regression via Posterior Sampling <https://arxiv.org/abs/2512.10849>`_

For more examples, see the ``demos/`` directory in the repository.
