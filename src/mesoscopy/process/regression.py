#  Copyright (c) 2022 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the
#   Software without restriction, including without limitation the rights to use, copy,
#   modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#   and to permit persons to whom the Software is furnished to do so, subject to the
#  following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies
#  or substantial portions of the Software
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
#  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
#  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
import os
from pathlib import Path

import click
import pandas as pd
from pynwb.image import ImageSeries
from skvideo.measure import mse
from mesoscopy import io
from mesoscopy import timer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


def ridge_regression(deltaf_series, regressors):
    """Perform ridge regression on a DeltaF/F series.

    Args:
        deltaf_series (np.ndarray): DeltaF/F series. Should be z-scored.
        regressors (np.ndarray): Regressor matrix.

    Returns:

    """
    regression_results = np.apply_along_axis(_pixel_ridge_regression, 0, deltaf_series, regressors)
    r2 = regression_results[-2, :]
    mse = regression_results[-1, :]
    coefficients = regression_results[:-2, :]

    return coefficients, r2, mse


def _pixel_ridge_regression(deltaf_series, regressors):
    """Perform ridge regression on a single pixel's DeltaF/F series.

    Args:
        deltaf_series (np.ndarray): DeltaF/F series for a single pixel. Should be z-scored.
        regressors (np.ndarray): Regressor matrix.

    Returns:
        np.ndarray: Array containing the regression coefficients, R^2 score, and mean squared error.
    """
    deltaf_series = np.nan_to_num(deltaf_series, nan=0.0)

    model = Ridge()
    model.fit(regressors, deltaf_series)

    r2 = r2_score(deltaf_series, model.predict(regressors))
    mse = mean_squared_error(deltaf_series, model.predict(regressors))

    return np.concatenate([model.coef_, [r2, mse]])
