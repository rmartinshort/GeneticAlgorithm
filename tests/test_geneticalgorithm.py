#!/usr/bin/env python

"""Tests for `geneticalgorithm` package."""

import os

import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from geneticalgorithm import GeneticAlgorithm


def test_integration():

    """

    :return:
    """

    feature_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "fixtures", "example.csv"))

    Y = pd.get_dummies(feature_df['activityID'], drop_first=True)

    X = feature_df.drop(['activityID'], axis=1)

    RF = RandomForestClassifier(n_estimators=1, min_samples_split=2, min_samples_leaf=1)

    GA = GeneticAlgorithm(X, Y, RF, njobs=4, Niter=10)

    GA.fit()

    assert (GA.best_fitness > 0.9)
    assert (len(GA.best_individual) == 302)
    assert (sum(GA.best_individual) > 0)
    assert (len(GA.best_individual_evolution) == 10)
    assert (np.array_equal(GA.best_individual_evolution[0],GA.best_individual_evolution[-1])==False)



@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
