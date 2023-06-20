import numpy as np

from scipy.optimize import minimize
from pymc.initial_point import make_initial_point_fn
from pymc.model import modelcontext
from pymc.util import get_default_varnames
from pytensor.compile.function import function
from pytensor.gradient import grad


__all__ = ['find_MLE']

class ModelContext:
    pass


class Fit:
    pass


class MCMC:
    pass

def find_MLE(
    model=None, start=None, method='L-BFGS-B', seed=42, return_raw=False,
    **kwargs
):
    model = modelcontext(model)
    vars_to_fit = model.continuous_value_vars
    neg_lnL = function(vars_to_fit, -model.datalogp)
    neg_lnL_grad = function(vars_to_fit, grad(-model.datalogp, vars_to_fit))
    vars_of_interest = get_default_varnames(model.unobserved_value_vars,
                                            include_transformed=False)
    get_voi = function(vars_to_fit, vars_of_interest)

    ipfn = make_initial_point_fn(
        model=model,
        jitter_rvs=set(),
        return_transformed=True,
        overrides=start,
    )
    start = ipfn(seed)
    model.check_start_vals(start)

    opt = minimize(fun=lambda x: neg_lnL(*x),
                   x0=[start[var.name] for var in vars_to_fit],
                   method=method,
                   jac=lambda x: neg_lnL_grad(*x),
                   **kwargs)

    voi_name = [p.name for p in vars_of_interest]
    voi_value = get_voi(*opt.x)
    voi = {name: value for name, value in zip(voi_name, voi_value)}
    voi['grad'] = np.linalg.norm(opt.jac)
    voi['stat'] = 2.0 * opt.fun

    if return_raw:
        return voi, opt
    else:
        return voi
