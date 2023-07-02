import pathlib
import numpy as np
import pytensor
import pytensor.tensor as pt
import xspec_models_cxc as _xsmodel

from pyda.bayespec.base_model import NumericGradOp

_xsmodel.chatter(0)

__all__ = list(_xsmodel.list_models())
all_models = _xsmodel.list_models()

_path = pathlib.Path(__file__).parent / 'xsmodel_class.py'
if not _path.exists():
    _template = """
class {mod_name}(SpectralModel):
    pars_name = {pars_name}
    pars_range = {pars_range}
    pars_default = {pars_default}
    pars_frozen = {pars_frozen}
    def __init__(self, {pars_expr}, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('{mod_name}', [{pars_list}], settings, grad_method, eps)
        super().__init__(op, op.optype)
    """
    _code = ''
    _code += 'from pyda.bayespec.xsmodel import XspecNumericGradOp\n'
    _code += 'from pyda.bayespec.base_model import SpectralModel\n'
    for _ in _xsmodel.list_models():
        pars_list = [p.name for p in _xsmodel.info(_).parameters]
        pars_range = [(p.hardmin, p.hardmax) for p in _xsmodel.info(_).parameters]
        pars_default = [p.default for p in _xsmodel.info(_).parameters]
        pars_frozen = [p.frozen for p in _xsmodel.info(_).parameters]
        pars_expr = pars_list + ['*']
        _code += '\n' + _template.format(mod_name=_,
                                         pars_name=str(tuple(pars_list)),
                                         pars_range=tuple(pars_range),
                                         pars_default=tuple(pars_default),
                                         pars_frozen=tuple(pars_frozen),
                                         pars_expr=', '.join(pars_expr),
                                         pars_list=', '.join(pars_list))
    with _path.open('w') as f:
        f.write(_code)
    del _code, _template

del _path


class XspecNumericGradOp(NumericGradOp):
    def __init__(self, modname, pars, settings='', grad_method='f', eps=1e-7):
        if modname not in _xsmodel.list_models():
            raise ValueError(f'Model "{modname}" not found')

        if settings:
            # for xsect and abund settings
            exec(settings)

        super().__init__(pars,
                         _xsmodel.info(modname).modeltype.name.lower(),
                         grad_method,
                         eps)

        xsfunc = getattr(_xsmodel, modname)
        language = _xsmodel.info(modname).language.name

        if pytensor.config.floatX == 'float64' and language == 'F77Style4':
            def _xsfunc(*args):
                return np.float64(xsfunc(*args))
        elif pytensor.config.floatX == 'float32' and language != 'F77Style4':
            def _xsfunc(*args):
                return np.float32(xsfunc(*args))
        else:
            _xsfunc = xsfunc

        # if modname not in _XSFUNC:
        #     _XSFUNC[modname] = _xsfunc
        # self._modname = modname

        self._xsfunc = _xsfunc

        self.itypes = [
            pt.TensorType('floatX', shape=())
            for _ in pars
        ]
        self.itypes.append(
            pt.TensorType('floatX', shape=(None,))  # ebins
        )

        if self.optype == 'con':
            self.itypes.append(
                pt.TensorType('floatX', shape=(None,))  # flux
            )

    def _perform(self, *inputs):
        pars = np.array(inputs[:self.npars])
        # return _XSFUNC[self._modname](pars, *inputs[self.npars:])
        return self._xsfunc(pars, *inputs[self.npars:])


from pyda.bayespec.xsmodel_class import *