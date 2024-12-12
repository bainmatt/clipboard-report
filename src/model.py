"""
Custom machine learning model transformers.

TODO: rename this to transform.py
"""

import holidays
import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """
    _summary_

    Parameters
    ----------
    n_clusters : int, optional
        _description_, by default 10
    gamma : float, optional
        _description_, by default 1.0
    random_state : int, optional
        Random state to pass to random number generator for reproducibility, by default None.

    References
    ----------
    .. [1] Géron, A. (2022). Hands-on machine learning with Scikit-Learn,
           Keras, and TensorFlow. " O'Reilly Media, Inc.".
    """

    def __init__(
        self,
        n_clusters: int = 10,
        gamma: float = 1.0,
        random_state: int | None = None
    ):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


class Detrender():
    """
    A custom regressor that removes the linear trend and seasonal component from a time series.

    Parameters
    ----------
    period : int, optional
        The seasonal period for decomposition, by default 52.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from src.model import Detrender
    >>> from src.stylesheet import customize_plots
    >>> customize_plots()

    Example time series data (dummy target variable)

    >>> np.random.seed(0)
    >>> time = np.arange(100)
    >>> trend = np.linspace(0, 1, 100)
    >>> seasonal = np.sin(time / 5) * 0.5
    >>> noise = np.random.normal(0, 0.1, 100)
    >>> y_train = trend + seasonal + noise

    Transform the target

    >>> transformer = Detrender(period=12)
    >>> y_transformed = transformer.fit_transform(y_train)
    >>> y_recovered = transformer.inverse_transform(y_transformed)

    Verify the results

    >>> print("Original y: ", y_train[:5].round(3))
    Original y:  [0.176 0.149 0.313 0.537 0.586]
    >>> print("Transformed y: ", y_transformed[:5].round(3))
    Transformed y:  [-0.035 -0.001  0.158  0.411  0.425]
    >>> print("Recovered y: ", y_recovered[:5].round(3).flatten())
    Recovered y:  [0.176 0.149 0.313 0.537 0.586]

    >>> message = "y_train and y_recovered do not match."
    >>> assert np.allclose(y_train[:10], y_recovered[:10].flatten()), message

    Get the trend and seasonal components for plotting

    >>> trend = transformer.trend_model.predict(
    ...     np.arange(len(y_train)).reshape(-1, 1)
    ... )
    >>> seasonal_component = transformer.decomposition.seasonal[:len(y_train)]

    Plot original data

    >>> plt.figure(figsize=(12, 8))  # doctest: +ELLIPSIS
    <...>
    >>> ax = plt.subplot(4, 1, 1)
    >>> ax.plot(y_train)  # doctest: +ELLIPSIS
    [...]
    >>> ax.set_title('Original Data')  # doctest: +ELLIPSIS
    Text...
    >>> ax.grid()

    Plot trend

    >>> ax = plt.subplot(4, 1, 2)
    >>> ax.plot(trend)  # doctest: +ELLIPSIS
    [...]
    >>> ax.set_title('Trend Component')  # doctest: +ELLIPSIS
    Text...
    >>> ax.grid()

    Plot seasonal component

    >>> ax = plt.subplot(4, 1, 3)
    >>> ax.plot(seasonal_component)  # doctest: +ELLIPSIS
    [...]
    >>> ax.set_title('Seasonal Component')  # doctest: +ELLIPSIS
    Text...
    >>> ax.grid()

    Plot Recovered Data

    >>> ax = plt.subplot(4, 1, 4)
    >>> ax.plot(y_recovered)  # doctest: +ELLIPSIS
    [...]
    >>> ax.set_title('Recovered Data')  # doctest: +ELLIPSIS
    Text...
    >>> ax.grid()
    >>> plt.show()

    Apply the learned transformation to test targets of a different length

    >>> y_test = np.sin(np.arange(50) / 5) * 0.5
    >>> y_test_transformed = transformer.transform(y_test)
    >>> y_test_recovered = transformer.inverse_transform(y_test_transformed)
    >>> assert np.allclose(y_test[:5], y_test_recovered[:5].flatten()), message
    """

    def __init__(self, period: int = 52):
        self.period = period
        self.trend_model = LinearRegression()
        self.decomposition = None
        self.time_index = None

    def fit(self, y):
        if y.ndim != 1:
            raise ValueError("Input y must be a one-dimensional array.")

        # Fit the trend model
        self.time_index = np.arange(len(y))
        self.trend_model.fit(self.time_index.reshape(-1, 1), y)

        # Decompose the seasonal component
        self.decomposition = seasonal_decompose(
            y, period=self.period, model='additive'
        )
        return self

    def transform(self, y):
        if self.time_index is None or self.decomposition is None:
            raise ValueError("The model must be fitted before transformation.")

        trend = self.trend_model.predict(np.arange(len(y)).reshape(-1, 1))

        # Detrend and deseasonalize
        y_detrended_deseasonalized = (
            y - trend - self.decomposition.seasonal[:len(y)]
        )
        return y_detrended_deseasonalized

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        if self.decomposition is None:
            raise ValueError(
                "The model must be fitted before inverse transformation."
            )

        trend = self.trend_model.predict(np.arange(len(y)).reshape(-1, 1))

        # Add back the trend and seasonal component
        y_recovered = (
            y.flatten() + trend + self.decomposition.seasonal[:len(y)]
        ).reshape(-1, 1)

        return y_recovered


class TimeFeatureAdder(BaseEstimator, TransformerMixin):
    """
    Add time features to a dataframe with a DateTime index.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the added time-based features:
        - `day_of_year`: Integer, the day of the year (1-365/366).
        - `is_weekend`': Boolean, True if the day is a weekend.
        - `is_holiday`: Boolean, True if the day is a public holiday in the US.

    Raises
    ------
    ValueError
        If the input DataFrame does not have a DateTime index.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from src.model import TimeFeatureAdder

    >>> date_range = pd.date_range(start='2023-12-21', end='2023-12-26')
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     index=date_range,
    ...     data={'target': np.random.rand(len(date_range))}
    ... )

    >>> time_feature_adder = TimeFeatureAdder()
    >>> transformed_data = time_feature_adder.fit_transform(data)

    >>> print(transformed_data.head())
                  target  is_weekend  is_holiday  day_of_year
    2023-12-21  0.374540       False       False          355
    2023-12-22  0.950714       False       False          356
    2023-12-23  0.731994        True       False          357
    2023-12-24  0.598658        True       False          358
    2023-12-25  0.156019       False        True          359
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # This transformer does not require fitting
        return self

    def transform(self, X, y=None):
        # Ensure X is a DataFrame and has a DateTime index
        is_dataframe = isinstance(X, pd.DataFrame)
        has_datetime_index = pd.api.types.is_datetime64_any_dtype(X.index)

        if not is_dataframe or not has_datetime_index:
            raise ValueError(
                "Input must be a pandas DataFrame with a DateTime index."
            )

        # Add time features
        df = X.copy()
        us_holidays = holidays.US()

        df['is_weekend'] = df.index.weekday >= 5
        df['is_holiday'] = df.index.map(lambda x: x in us_holidays)
        df['day_of_year'] = df.index.dayofyear

        return df


def main():
    import doctest
    doctest.testmod(verbose=True)

    # from src.workflow import doctest_function
    # doctest_function(Detrender, globs=globals())

    # -- One-off tests -------------------------------------------------------

    # from src.inspection import display
    # y = np.sin(np.arange(50) / 5) * 0.5
    # display("y", "np.arange(len(y)).reshape(-1, 1)", globs=globals())

    pass


if __name__ == "__main__":
    from src.stylesheet import customize_plots
    customize_plots()

    main()
