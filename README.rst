================
GeneticAlgorithm
================


.. image:: https://img.shields.io/pypi/v/geneticalgorithm.svg
        :target: https://pypi.python.org/pypi/geneticalgorithm

.. image:: https://img.shields.io/travis/rmartinshort/geneticalgorithm.svg
        :target: https://travis-ci.com/rmartinshort/geneticalgorithm

.. image:: https://readthedocs.org/projects/geneticalgorithm/badge/?version=latest
        :target: https://geneticalgorithm.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


GeneticAlgorithm
--------

A genetic algorithm for selecting optimal feature combinations with sklearn supervised ML objects.

See the docstring of GeneticAlgorithm.py for a brief explanation for how this works.

Useage
--------

* This should work with any Sklearn classifer or regression object. It can be used as in this simple example

First, make a classifier of your choice, and select hyperparameters. Note that the generic algorithm does not select hyperparameters, only features to train on.

* RF = RandomForestClassifier(n_estimators=1,min_samples_split=2,min_samples_leaf=1)

Make a genetic algorithm object. X and Y here are dataframes of the predictors (X) and target vector (Y).

* GA = GeneticAlgorithm(X,Y,RF,njobs=4)

Call GA.fit(). This will run the algorithm. It may take a long time!

* GA.fit()

An optimal combination of columns is selected and this can then be used in future developments. The selected features are
assigned to the variable 'GA.feature_selection'


* Free software: MIT license
* Documentation: https://geneticalgorithm.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
