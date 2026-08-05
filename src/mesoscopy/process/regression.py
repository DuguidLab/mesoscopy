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

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def ridge_regression(deltaf_series: np.ndarray, regressors: np.ndarray) -> tuple:
    """Pixel-wise ridge regression on a (time x height x width) deltaF series.

    Assumes that the deltaF series has been z-scored along the time axis.

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


def ridge_regression_fast(deltaf_series: np.ndarray, regressors: np.ndarray, alpha: float = 1.0) -> tuple:
    """Vectorised implementation of pixel-wise ridge regression on a (time x height x width) deltaF series.

    Fits a separate ridge regression model (with intercept) for every pixel's time series
    at once, using a vectorised closed-form solution instead of looping over pixels.
    This is much faster than the naive implementation, especially for large images.

    Args:
        deltaf_series (np.ndarray): DeltaF/F series. Should be z-scored. Shape is
            (n_samples, x, y).
        regressors (np.ndarray): Regressor matrix of shape (n_samples, n_regressors).
        alpha (float): Ridge regularisation strength. Defaults to 1.0, matching
            ``sklearn.linear_model.Ridge``'s default.

    Returns:
        tuple: (coefficients, r2, mse), where coefficients has shape
        (n_regressors, *pixel_shape), and r2/mse have shape (*pixel_shape,).
    """
    pixel_shape = deltaf_series.shape[1:]
    n_samples = deltaf_series.shape[0]

    # Cast everything to float32 to reduce memory usage and speed up computation.
    dtype = np.result_type(deltaf_series.dtype, np.float32)

    # Reshape deltaf_series to (n_samples, n_pixels) for vectorised computation.
    y = np.nan_to_num(deltaf_series, nan=0.0).reshape(n_samples, -1).astype(dtype, copy=False)
    x = np.asarray(regressors, dtype=dtype)

    # Compute means and centre the data.
    x_mean = x.mean(axis=0, dtype=dtype)
    y_mean = y.mean(axis=0, dtype=dtype)
    x_centred = x - x_mean
    y_centred = y - y_mean

    n_regressors = x.shape[1]
    xtx = x_centred.T @ x_centred
    gram = xtx.copy()
    gram.flat[:: n_regressors + 1] += alpha
    xty = x_centred.T @ y_centred
    coef = np.linalg.solve(gram, xty)

    # Compute the residual/total sums of squares algebraically instead of forming a full
    # (n_samples, n_pixels) predictions/residuals array.
    ss_tot = np.einsum("tp,tp->p", y_centred, y_centred)
    cross_term = np.einsum("fp,fp->p", coef, xty)
    pred_sq = np.einsum("fp,fp->p", coef, xtx @ coef)
    # Clip tiny negative values arising from floating point cancellation.
    ss_res = np.clip(ss_tot - 2 * cross_term + pred_sq, 0, None)

    # Replicate sklearn.metrics.r2_score's handling of zero-variance targets.
    r2 = np.ones_like(ss_tot)
    nonzero_denominator = ss_tot != 0
    nonzero_numerator = ss_res != 0
    valid_score = nonzero_denominator & nonzero_numerator
    r2[valid_score] = 1 - (ss_res[valid_score] / ss_tot[valid_score])
    r2[nonzero_numerator & ~nonzero_denominator] = 0.0

    mse = ss_res / n_samples

    # Reshape the coefficients, r2, and mse back to the original pixel shape.
    coefficients = coef.reshape((n_regressors,) + pixel_shape)
    r2 = r2.reshape(pixel_shape)
    mse = mse.reshape(pixel_shape)

    return coefficients, r2, mse


def _pixel_ridge_regression(deltaf_series: np.ndarray, regressors: np.ndarray) -> np.ndarray:
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
