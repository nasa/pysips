API Reference
=============

This section contains the complete API documentation for PySIPS, automatically generated from the source code docstrings.

Main Components
---------------

The primary interface to PySIPS is through the :class:`~pysips.PysipsRegressor` class, which provides a scikit-learn compatible API for symbolic regression.

PysipsRegressor
~~~~~~~~~~~~~~~

.. autoclass:: pysips.PysipsRegressor
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Core Modules
------------

These modules provide the underlying implementation of the symbolic regression algorithm.

Sampler
~~~~~~~

.. automodule:: pysips.sampler
   :members:
   :undoc-members:
   :show-inheritance:

Prior
~~~~~

.. automodule:: pysips.prior
   :members:
   :undoc-members:
   :show-inheritance:

Metropolis
~~~~~~~~~~

.. automodule:: pysips.metropolis
   :members:
   :undoc-members:
   :show-inheritance:

Laplace NMLL
~~~~~~~~~~~~

.. automodule:: pysips.laplace_nmll
   :members:
   :undoc-members:
   :show-inheritance:

Proposal Mechanisms
-------------------

Mutation Proposal
~~~~~~~~~~~~~~~~~

.. automodule:: pysips.mutation_proposal
   :members:
   :undoc-members:
   :show-inheritance:

Crossover Proposal
~~~~~~~~~~~~~~~~~~

.. automodule:: pysips.crossover_proposal
   :members:
   :undoc-members:
   :show-inheritance:

Random Choice Proposal
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pysips.random_choice_proposal
   :members:
   :undoc-members:
   :show-inheritance:
