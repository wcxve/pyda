import numpy as np
import scipy.stats as stats
from scipy.special import gammainc, gammaincc, gammaincinv


def upper_limit(n, b, cl):
    """Calculate upper limit.

    Parameters
    ----------
    n : array_like
        Observed counts.
    b : array_like
        Estimated background counts.
    cl : float or int, optional
        Confidence level for the confidence interval. If 0 < `cl` < 1, the
        value is interpreted as the confidence level. If `cl` >= 1, it is
        interpreted as the number of standard deviations. For example,
        ``cl=1`` produces a 1-sigma or 68.3% confidence interval.

    Returns
    -------
    ul : ndarray
        The upper limit of the signal counts.
    """
    n = np.array(n)
    b = np.array(b)
    cl = float(cl)
    assert cl > 0.0, 'cl must be positive'
    cl_ = 1.0 - 2.0 * stats.norm.sf(cl) if cl >= 1.0 else cl
    a = n + 1.0
    return gammaincinv(a, cl_ * gammaincc(a, b) + gammainc(a, b)) - b


if __name__ == '__main__':
    print(upper_limit(10000, 10000, 0.997))
