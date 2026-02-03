Quickstart Guide
================

This quickstart guide will help you get up and running with PySIPS in under 5 minutes.

Installation
------------

Install PySIPS using pip:

.. code-block:: bash

   pip install pysips

Basic Usage
-----------

Here's a minimal example to get you started with symbolic regression:

.. code-block:: python

   import numpy as np
   from pysips import PysipsRegressor
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import r2_score

   # Generate synthetic data: y = x^2 + noise
   np.random.seed(42)
   X = np.linspace(-3, 3, 100).reshape(-1, 1)
   y = X[:, 0]**2 + np.random.normal(0, 0.1, size=X.shape[0])

   # Split into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # Create the regressor
   regressor = PysipsRegressor(
       operators=['+', '-', '*'],  # Available operators
       max_complexity=12,          # Maximum expression size
       num_particles=100,          # Population size
       num_mcmc_samples=10,        # MCMC steps per iteration
       max_time=60,                # Maximum runtime in seconds
       random_state=42
   )

   # Fit the regressor
   regressor.fit(X_train, y_train)

   # Make predictions
   y_pred = regressor.predict(X_test)

   # Get the discovered expression
   expression = regressor.get_expression()
   print(f"Discovered expression: {expression}")
   print(f"R² score: {r2_score(y_test, y_pred):.4f}")

Expected Output
---------------

.. code-block:: text

   Discovered expression: x_0^2
   R² score: 0.9987

Understanding the Parameters
-----------------------------

**Essential Parameters:**

- ``operators``: List of mathematical operators to use (e.g., ``['+', '-', '*', '/', 'pow']``)
- ``max_complexity``: Maximum size of the expression graph (controls model complexity)
- ``num_particles``: Number of particles in the SMC population (higher = better exploration)
- ``random_state``: Random seed for reproducibility

**Time Control:**

- ``max_time``: Maximum runtime in seconds (default: no limit)
- ``show_progress_bar``: Display progress during fitting (default: True)

**Model Selection:**

- ``model_selection``: Choose ``'mode'`` (most frequent) or ``'max_likelihood'`` (best scoring)

Accessing Results
-----------------

After fitting, you can access various results:

.. code-block:: python

   # Get the best expression as a string
   expression = regressor.get_expression()

   # Get all unique models and their likelihoods
   models, likelihoods = regressor.get_models()
   print(f"Number of unique models: {len(models)}")

   # Make predictions on new data
   y_pred = regressor.predict(X_new)

Next Steps
----------

- Continue to the :doc:`tutorial_notebook` for more advanced usage and examples
- Explore the :doc:`api/modules` for detailed API documentation
- Check out the examples in the ``demos/`` directory of the repository

Common Issues
-------------

**Long Runtime:**

If fitting takes too long, try:
- Reducing ``num_particles`` (e.g., 50-100 for quick experiments)
- Reducing ``max_complexity`` (e.g., 10-15 for simpler expressions)
- Setting ``max_time`` to limit the runtime

**Poor Results:**

If the discovered expression is not accurate, try:
- Increasing ``num_particles`` (e.g., 200-500 for better exploration)
- Adjusting the available ``operators``
- Increasing ``max_complexity`` if you expect more complex relationships
- Running for longer (increase or remove ``max_time``)
