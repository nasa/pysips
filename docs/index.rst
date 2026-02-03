.. PySIPS documentation master file

PySIPS: Python package for Symbolic Inference via Posterior Sampling
=====================================================================

Welcome to PySIPS documentation! PySIPS is a robust framework for discovering interpretable symbolic expressions from data using Bayesian symbolic regression via Sequential Monte Carlo (SMC) sampling.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   tutorial_notebook

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

Overview
--------

PySIPS provides a Bayesian approach to symbolic regression that offers several advantages over traditional methods:

* **Robustness to noise**: The Bayesian framework naturally handles noisy datasets
* **Uncertainty quantification**: Provides built-in uncertainty estimates for discovered expressions
* **Parsimonious expressions**: Discovers simple, interpretable models with improved generalization
* **Reduced overfitting**: Sequential Monte Carlo sampling mitigates overfitting in symbolic regression

The algorithm uses Sequential Monte Carlo (SMC) sampling to approximate the posterior distribution over symbolic expressions, employing normalized marginal likelihood for model evaluation and combining mutation and crossover operations as proposal mechanisms.

Key Features
------------

* **Scikit-learn compatible API**: Easy integration into existing ML pipelines
* **Flexible operator selection**: Control over mathematical operators and expression complexity
* **Multiple model selection strategies**: Choose from mode or maximum likelihood selection
* **Checkpoint support**: Save and resume long-running symbolic regression tasks
* **Access to full posterior**: Examine the complete distribution over candidate expressions

Installation
------------

Install PySIPS using pip:

.. code-block:: bash

   pip install pysips

Quick Example
-------------

.. code-block:: python

   import numpy as np
   from pysips import PysipsRegressor

   # Generate sample data
   X = np.linspace(-3, 3, 100).reshape(-1, 1)
   y = X[:, 0]**2 + np.random.normal(0, 0.1, size=X.shape[0])

   # Create and fit the regressor
   regressor = PysipsRegressor(
       operators=['+', '-', '*'],
       max_complexity=12,
       num_particles=100,
       random_state=42
   )
   regressor.fit(X, y)

   # Get the discovered expression
   expression = regressor.get_expression()
   print(f"Discovered expression: {expression}")

.. note::
   For an interactive tutorial with code you can run and modify, see the :doc:`tutorial_notebook` or download the `tutorial notebook <tutorial_notebook.ipynb>`_ to run locally.

Citation
--------

If you use PySIPS in your research, please cite:

.. code-block:: bibtex

   @article{bomarito2024bayesian,
     title={Bayesian Symbolic Regression via Posterior Sampling},
     author={Bomarito, Geoffrey F. and Leser, Patrick E.},
     journal={Philosophical Transactions of the Royal Society A},
     year={2025},
     publisher={Royal Society}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
