import numpy as np


def KSstat(data, model):
    """
    Kolmogorov-Smirnov statistic: maximum deviance between data and model
    """
    modelc = model.cumsum() / model.sum()
    datac = data.cumsum() / data.sum()
    ks = np.abs(modelc - datac).max()
    return ks


def CvMstat(data, model):
    """
    Cramér–von Mises statistic: Takes all deviances into account
    """
    modelc = model.cumsum()
    datac = data.cumsum()
    maxmodelc = modelc.max()
    cvm = ((
                   modelc / maxmodelc - datac / datac.max()) ** 2 * model / maxmodelc).mean()
    return cvm


def ADstat(data, model):
    """
    Anderson-Darling statistic: Takes all deviances into account
    more weight on tails than CvM.
    """
    modelc = model.cumsum()
    datac = data.cumsum()
    maxmodelc = modelc.max()
    valid = np.logical_and(modelc > 0, maxmodelc - modelc > 0)
    modelc = modelc[valid] / maxmodelc
    datac = datac[valid] / datac.max()
    model = model[valid] / maxmodelc
    assert (modelc > 0).all(), ['ADstat has zero cumulative denominator',
                                modelc]
    assert (maxmodelc - modelc > 0).all(), [
        'ADstat has zero=1-1 cumulative denominator', maxmodelc - modelc]
    ad = ((modelc - datac) ** 2 / (
            modelc * (maxmodelc - modelc)) * model).sum()
    return ad
