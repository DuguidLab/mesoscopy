#  Copyright (c) 2026 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
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
import numpy.typing as npt
from scipy.stats import norm
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def logistic_decoder(deltaf_series: np.ndarray, labels: np.ndarray, test_size: float = 0.2) -> dict:
    """Logistic regression decoder for a (time x height x width) deltaF series.

    Args:
        deltaf_series (np.ndarray): DeltaF/F series. Should be z-scored. Shape is
            (n_samples, x, y), or (n_samples, n_pixels) if the spatial dimensions have already
            been masked/flattened.
        labels (np.ndarray): Labels for each sample, ie variable to be decoded. Should be a binary array.
            Shape is (n_samples,).
        test_size (float): Fraction of the dataset to include in the test split.

    Returns:
        dict: Dictionary containing evaluation metrics and model coefficients. Coefficients are
            returned with the same shape as the input's non-time dimensions.
    """
    data = deltaf_series.reshape(deltaf_series.shape[0], -1)  # Flatten spatial dimensions
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=0)

    model = LogisticRegression(solver="lbfgs")
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    conf_counts = metrics.confusion_matrix(y_test, predictions)

    accuracy = metrics.accuracy_score(y_test, predictions)
    balanced_accuracy = metrics.balanced_accuracy_score(y_test, predictions)
    coefs = model.coef_.reshape(deltaf_series.shape[1:])
    conf_matrix = conf_counts / conf_counts.sum(axis=1, keepdims=True)
    d_prime = _d_prime(conf_counts)
    f2_score = metrics.fbeta_score(y_test, predictions, beta=2)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "d_prime": d_prime,
        "confusion_matrix": conf_matrix,
        "test_size": test_size,
        "f2_score": f2_score,
        "coefficients": coefs,
    }


def lda_decoder(deltaf_series: np.ndarray, labels: np.ndarray, test_size: float = 0.2) -> dict:
    """Linear Discriminant Analysis (LDA) decoder for a (time x height x width) deltaF series.

    Args:
        deltaf_series (np.ndarray): DeltaF/F series. Should be z-scored. Shape is
            (n_samples, x, y), or (n_samples, n_pixels) if the spatial dimensions have already
            been masked/flattened.
        labels (np.ndarray): Labels for each sample, ie variable to be decoded. Should be a binary array.
            Shape is (n_samples,).
        test_size (float): Fraction of the dataset to include in the test split.

    Returns:
        dict: Dictionary containing evaluation metrics and model coefficients. Coefficients are
            returned with the same shape as the input's non-time dimensions.
    """
    data = deltaf_series.reshape(deltaf_series.shape[0], -1)  # Flatten spatial dimensions
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=0)

    model = LinearDiscriminantAnalysis()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    conf_counts = metrics.confusion_matrix(y_test, predictions)

    accuracy = metrics.accuracy_score(y_test, predictions)
    balanced_accuracy = metrics.balanced_accuracy_score(y_test, predictions)
    coefs = model.coef_.reshape(deltaf_series.shape[1:])
    conf_matrix = conf_counts / conf_counts.sum(axis=1, keepdims=True)
    d_prime = _d_prime(conf_counts)
    f2_score = metrics.fbeta_score(y_test, predictions, beta=2)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "d_prime": d_prime,
        "confusion_matrix": conf_matrix,
        "test_size": test_size,
        "f2_score": f2_score,
        "coefficients": coefs,
    }


def _d_prime(conf_matrix: npt.NDArray) -> float:
    """Sensitivity index (d') for a binary classification, from a confusion matrix of counts.

    Perfect (or perfectly inverted) classification yields hit and false alarm rates of exactly 1 and
    0, whose z-transforms are infinite. Such extreme rates are replaced by 1 - 1/(2N) and 1/(2N)
    respectively, where N is the number of test samples of the corresponding true class
    (Macmillan & Kaplan, 1985), keeping d' finite and bounded by the size of the test set.

    Args:
        conf_matrix (npt.NDArray): A 2x2 confusion matrix of sample counts, with the true classes
            along the rows and the predicted classes along the columns.

    Returns:
        float: The sensitivity index d'.
    """
    class_counts = conf_matrix.sum(axis=1, keepdims=True)
    # Any non-extreme rate already lies within [1/(2N), 1 - 1/(2N)], so this only touches 0s and 1s.
    correction = 1 / (2 * class_counts)
    rates = np.clip(conf_matrix / class_counts, correction, 1 - correction)

    return float(norm.ppf(rates[0, 0]) - norm.ppf(rates[1, 0]))
