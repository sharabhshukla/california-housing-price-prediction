import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


class DataConfig:
    columns = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]


class CaliforniaPricingModel:
    """
    This class loads up a pre trained gradient boosting
    model on california pricing dataset.
    methods for batch and single predictions are also available
    """

    def __init__(
        self,
        preprocess_model="models/preprocessor",
        regressor="models/catboost_regressor",
    ):
        self._preprocessor = joblib.load(preprocess_model)
        self._regressor = CatBoostRegressor()
        self._regressor.load_model(regressor)

    def batch_prediction(self, x_data: pd.DataFrame):
        """
        This functions is for batch predictions, takes in an array of
        (sample_size*features), returns a
        array of predicted prices
        Parameters
        ----------
        x_data: pd.DataFrame

        Returns
        -------
        predicted_price: np.array
        """
        predicted_price = np.expm1(
            self._regressor.predict(self._preprocessor.transform(x_data))
        )
        return predicted_price

    def price_prediction(
        self,
        longitude: float,
        latitude: float,
        housing_median_age: float,
        total_rooms: float,
        total_bedrooms: float,
        population: float,
        households: float,
        median_income: float,
    ) -> float:
        """
        This function returns a single prediction of the house with given parameters
        Parameters
        ----------
        longitude: float
        latitude: float
        housing_median_age: float
        total_rooms: float
        total_bedrooms: float
        population: float
        households: float
        median_income: float

        Returns
        -------
        predicted_price: float
        """

        x_data = np.array(
            [
                longitude,
                latitude,
                housing_median_age,
                total_rooms,
                total_bedrooms,
                population,
                households,
                median_income,
            ]
        ).reshape(1, len(DataConfig.columns))
        x_data_df = pd.DataFrame(x_data, columns=DataConfig.columns)
        predicted_price = np.expm1(
            self._regressor.predict(self._preprocessor.transform(x_data_df))
        )
        return predicted_price
