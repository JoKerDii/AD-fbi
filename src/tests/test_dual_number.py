import pytest
import numpy as np
from fbi.dual_number import dual_number, var_type

z0 = dual_number(1, 2)
z1 = dual_number(1, -1)
z2 = dual_number(2.2, 0)
z3 = dual_number(2, 6.036)
z4 = dual_number(3, 100000)
z5 = dual_number(0.5, -2)

class TestDualNumber:
    """Test class for dual_number module"""
    
    def test_var_type(self):
        x1, x2, x3, x4 = 10, 'a', 'deriv', ['d', 'e', 'r', 'i', 'v']
        x5, x6, x7, x8 = [1, 'e', 2, 'i', 'v'], [1, 2, 3, 4, 5], \
                                   ["test", 3, 0, 'c'], [0.1, 0.2, -10]

        assert var_type(x1) == True
        assert var_type(x2) == False
        assert var_type(x3) == False
        assert var_type(x4) == False
        assert var_type(x5) == False
        assert var_type(x6) == True
        assert var_type(x7) == False
        assert var_type(x8) == True
        
    def test_init(self):
        assert z0.val == 1
        assert z0.derv == 2
    
    def test_add(self):
        out = z1 + z2
        assert out.val == 3.2
        assert out.derv == -1
        out = z1 + 2
        assert out.val == 3
        assert out.derv == -1
        out = 2 + z1
        assert out.val == 3
        assert out.derv == -1
        
    def test_sub(self):
        out = z1 - z0
        assert out.val == 0
        assert out.derv == -3
        out = z1 - 2
        assert out.val == -1
        assert out.derv == -1
        out = 2 - z1
        assert out.val == 1
        assert out.derv == 1
        
    def test_mul(self):
        out = z1 * z2
        assert out.val == 2.2
        assert out.derv == -2.2
        out = z1 * 2
        assert out.val == 2
        assert out.derv == -2
        out = 2 * z1
        assert out.val == 2
        assert out.derv == -2
        
    def test_div(self):
        out = z1 / z3
        assert out.val == 0.5
        assert out.derv == -8.036 / 4
        out = z1 / 2
        assert out.val == 0.5
        assert out.derv == -0.5
        out = 2 / z1
        assert out.val == 2
        assert out.derv == 2
        
    def test_pow(self):
        out = z5 ** z4
        assert out.val == 0.5 ** 3
        assert out.derv == (0.5 ** 3) * (100000 * np.log(0.5) + (-2) * 3 / 0.5)
        out = z4 ** 2
        assert out.val == 3 ** 2
        assert out.derv == (3 ** 2) * (100000 * 2 / 3)
        out = 2 ** z4
        assert out.val == 2 ** 3
        assert out.derv == (2 ** 3) * (100000 * np.log(2))
        
    def test_neg(self):
        out = -z1
        assert out.val == -1
        assert out.derv == 1
        
    def test_eq(self):
        tmp = dual_number(1,-1)
        assert tmp.val == z1.val
        assert tmp.derv == z1.derv
