import pytest
import numpy as np
from fbi.dual_number import DualNumbers, is_numeric
from fbi.forward_mode import ForwardMode


def test_univariate_scalar_f(self):
    func = lambda x: 2 * x + x ** 2 + x.log() + 1 / x
    func_val = lambda x: 2 * x + x ** 2 + np.log(x) + 1 / x
    func_derv = lambda x: 2 + 2 * x + 1 / x - 1 / (x ** 2)

    ad = ForwardMode(2, func)

    f_val = ad.get_fx_value()
    f_derv = ad.get_derivative()

    self.assertAlmostEqual(f_val, func_val(2))
    self.assertAlmostEqual(f_derv, func_derv(2))
