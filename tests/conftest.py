# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for california_housing_price.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import pytest

import pandas as pd
from sklearn.metrics import r2_score

from california_housing_price.california_pricing_model import CaliforniaPricingModel


@pytest.fixture
def test_setup():
    model = CaliforniaPricingModel()
    return model


@pytest.fixture
def test_df_setup():
    test_data = pd.read_csv("data/external/california_housing_test.csv")
    return test_data


def test_price_prediction(test_setup):
    house_features = [
        -1.2205e02,
        3.7370e01,
        2.7000e01,
        3.8850e03,
        6.6100e02,
        1.5370e03,
        6.0600e02,
        6.6085e00,
    ]
    price_prediction = 441451.25438585
    price = test_setup.price_prediction(*house_features)
    assert price == pytest.approx(price_prediction)


def test_batch_prediction(test_setup, test_df_setup):
    test_data = test_df_setup
    x_data = test_data.drop(columns="median_house_value", axis=1)
    y_data = test_data["median_house_value"]
    model = test_setup
    y_preds = model.batch_prediction(x_data)
    assert 0.8277017351032162 == pytest.approx(r2_score(y_data, y_preds), 1e-3)
