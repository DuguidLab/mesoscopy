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
from scipy.stats import norm
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def logistic_decoder(deltaf_series: np.ndarray, labels: np.ndarray, test_size: float = 0.2) -> dict:
    """Logistic regression decoder for a (time x height x width) deltaF series.

    Args:
        deltaf_series (np.ndarray): DeltaF/F series. Should be z-scored. Shape is
            (n_samples, x, y).
        labels (np.ndarray): Labels for each sample, ie variable to be decoded. Should be a binary array.
            Shape is (n_samples,).
        test_size (float): Fraction of the dataset to include in the test split.

    Returns:
        dict: Dictionary containing evaluation metrics and model coefficients.
    """
    data = deltaf_series.reshape(deltaf_series.shape[0], -1)  # Flatten spatial dimensions
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=0)

    model = LogisticRegression(solver="lbfgs")
    model.fit(x_train, y_train)

    accuracy = metrics.accuracy_score(y_test, model.predict(x_test))
    balanced_accuracy = metrics.balanced_accuracy_score(y_test, model.predict(x_test))
    coefs = model.coef_.reshape(deltaf_series.shape[1], deltaf_series.shape[2])
    conf_matrix = metrics.confusion_matrix(y_test, model.predict(x_test), normalize="true")
    d_prime = norm.ppf(conf_matrix[0, 0]) - norm.ppf(conf_matrix[1, 0])
    f2_score = metrics.fbeta_score(y_test, model.predict(x_test), beta=2)

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
            (n_samples, x, y).
        labels (np.ndarray): Labels for each sample, ie variable to be decoded. Should be a binary array.
            Shape is (n_samples,).
        test_size (float): Fraction of the dataset to include in the test split.

    Returns:
        dict: Dictionary containing evaluation metrics and model coefficients.
    """
    data = deltaf_series.reshape(deltaf_series.shape[0], -1)  # Flatten spatial dimensions
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=0)

    model = LinearDiscriminantAnalysis()
    model.fit(x_train, y_train)

    accuracy = metrics.accuracy_score(y_test, model.predict(x_test))
    balanced_accuracy = metrics.balanced_accuracy_score(y_test, model.predict(x_test))
    coefs = model.coef_.reshape(deltaf_series.shape[1], deltaf_series.shape[2])
    conf_matrix = metrics.confusion_matrix(y_test, model.predict(x_test), normalize="true")
    d_prime = norm.ppf(conf_matrix[0, 0]) - norm.ppf(conf_matrix[1, 0])
    f2_score = metrics.fbeta_score(y_test, model.predict(x_test), beta=2)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "d_prime": d_prime,
        "confusion_matrix": conf_matrix,
        "test_size": test_size,
        "f2_score": f2_score,
        "coefficients": coefs,
    }
