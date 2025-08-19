#  Copyright (c) 2025 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
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
import scipy.ndimage as ndi


def laplace_gaussian(
    deltaf_series: np.ndarray,
    sigma: int = 2,
    **kwargs,
):
    """Spatial smoothing using a Laplace of Gaussian filter.

    Args:
        deltaf_series (np.ndarray): DeltaF/F series.
        sigma (int, optional): Standard deviation for Gaussian kernel. Defaults to 2.
        **kwargs: Additional keyword arguments for `scipy.ndimage.gaussian_laplace`.

    Returns:
        np.ndarray: Smoothed DeltaF/F series.
    """
    sigma_iso = (0, sigma, sigma)

    return ndi.gaussian_laplace(deltaf_series, sigma=sigma_iso, **kwargs)
