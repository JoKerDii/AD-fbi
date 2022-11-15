import pytest
import numpy as np
from fbi.dual_number import DualNumbers, is_numeric

z0 = DualNumbers(1, 2)
z1 = DualNumbers(1, -1)
z2 = DualNumbers(2.2, 0)
z3 = DualNumbers(2, 6.036)
z4 = DualNumbers(3, 100000)
z5 = DualNumbers(0.5, -2)
z6 = DualNumbers(0, 9)

x1 = DualNumbers(1, np.array([0, 2]))
x2 = DualNumbers(1, np.array([-1, 0]))
x3 = DualNumbers(1, np.array([0, -1]))
x4 = DualNumbers(0, np.array([-2, -1]))


class TestDualNumbers:
    """Test class for DualNumbers module"""
    
    # test is_numeric function
    def test_is_numeric(self):
        x1, x2, x3, x4 = 6, 'a', 'dual', ['n', 'u', 'm', 'b', 'e', 'r']
        x5, x6, x7, x8 = [3, 's', 9, 'i', 'x'], [1, 2, 3, 4, 5], \
                                   ["test", 2, 7, 'c'], [0.2, 0.5, -6]

        assert is_numeric(x1) == True
        assert is_numeric(x2) == False
        assert is_numeric(x3) == False
        assert is_numeric(x4) == False
        assert is_numeric(x5) == False
        assert is_numeric(x6) == True
        assert is_numeric(x7) == False
        assert is_numeric(x8) == True
        
    # test attribute initialization
    def test_init(self):
        assert z0.val == 1
        assert z0.derv == 2

        assert x1.val == 1
        assert x1.derv[0] == 0
        assert x1.derv[1] == 2
    # def test_init_Error(self):
    #     with self.assertRaises(ZeroDivisionError) as e:
    #         var1 / 0
    #     self.assertEqual('ERROR: Denominator in division should not be 0', str(e.exception))

    #     with self.assertRaises(ZeroDivisionError) as e:
    #         var1 / var6
    #     self.assertEqual('ERROR: Denominator in division should not be 0', str(e.exception))
        
    # test attribute setter
    def test_setter(self):
        z0.val = 2
        z0.derv = 1
        assert z0.val == 2
        assert z0.derv == 1
    
    # test scalar addition operation
    def test_scalar_add(self):
        out = z1 + z2
        assert out.val == 3.2
        assert out.derv == -1
        out = z1 + 2
        assert out.val == 3
        assert out.derv == -1
        out = 2 + z1
        assert out.val == 3
        assert out.derv == -1
    
    # test array addition operation
    def test_1d_add(self):
        out = x1 + x2
        assert out.val == 2 
        assert all(out.derv == [-1, 2])
        out = x1 + 2
        assert out.val == 3
        assert all(out.derv == [0, 2])
        out = 2 + x1
        assert out.val == 3
        assert all(out.derv == [0, 2])
    
    # test scalar subtraction operation
    def test_scalar_sub(self):
        out = z1 - z2
        assert out.val == 1-2.2
        assert out.derv == -1
        out = z1 - 2
        assert out.val == -1
        assert out.derv == -1
        out = 2 - z1
        assert out.val == 1
        assert out.derv == 1
    
    # test array subtraction operation
    def test_1d_sub(self):
        out = x2 - x1
        assert out.val == 0
        assert all(out.derv == [-1, -2])
        out = x1 - 2
        assert out.val == -1
        assert all(out.derv == [0, 2])
        out = 2 - x1
        assert out.val == 1
        assert all(out.derv == [0, -2])
    
    # test scalar multiplization operation
    def test_scalar_mul(self):
        out = z1 * z2
        assert out.val == 2.2
        assert out.derv == -2.2
        out = z1 * 2
        assert out.val == 2
        assert out.derv == -2
        out = 2 * z1
        assert out.val == 2
        assert out.derv == -2
        
    # test array multiplication operation
    def test_1d_mul(self):
        out = x1 * x2
        assert out.val == 1
        assert all(out.derv == [-1, 2])
        out = x1 * 2
        assert out.val == 2
        assert all(out.derv == [0, 4])
        out = 2 * x1
        assert out.val == 2
        assert all(out.derv == [0, 4])

    # test scalar division operation
    def test_scalar_div(self):
        out = z1 / z3
        assert out.val == 0.5
        assert out.derv == -8.036 / 4
        out = z1 / 2
        assert out.val == 0.5
        assert out.derv == -0.5
        out = 2 / z1
        assert out.val == 2
        assert out.derv == 2
        
    # test array division operation
    def test_1d_div(self):
        out = x2 / x1
        assert out.val == 1
        assert all(out.derv == [-1, -2])
        out = x1 / 2
        assert out.val == 0.5
        assert all(out.derv == [0, 1])
        out = 2 / x1
        assert out.val == 2
        assert all(out.derv == [0, -4])
    
    # test error handling in scalar division operation
    def test_scalar_div_zeroError(self):
        with pytest.raises(ZeroDivisionError) as e:
            z1 / 0
        with pytest.raises(ZeroDivisionError) as e:
            z1 / z6
        with pytest.raises(ZeroDivisionError) as e:
            4 / z6
    
    # test error handling in array division operation
    def test_1d_div_zeroError(self):
        with pytest.raises(ZeroDivisionError) as e:
            x2 / 0
        with pytest.raises(ZeroDivisionError) as e:
            x2 / x4
        with pytest.raises(ZeroDivisionError) as e:
            4 / x4
    
    # test scalar power operation
    def test_scalar_pow(self):
        out = z5 ** z4
        assert out.val == 0.5 ** 3
        assert out.derv == (0.5 ** 3) * (100000 * np.log(0.5) + (-2) * 3 / 0.5)
        out = z4 ** 2
        assert out.val == 3 ** 2
        assert out.derv == (3 ** 2) * (100000 * 2 / 3)
        out = 2 ** z4
        assert out.val == 2 ** 3
        assert out.derv == (2 ** 3) * (100000 * np.log(2))
        
    # test array power operation
    def test_1d_pow(self):
        out = x2 ** x1
        assert out.val == 1
        assert all(out.derv == [-1, 0])
        
    # test scalar negation operation
    def test_scalar_neg(self):
        out = -z1
        assert out.val == -1
        assert out.derv == 1
        
    # test array negation operation
    def test_1d_neg(self):
        out = -x1
        assert out.val == -1
        assert all(out.derv == [0, -2])
        
    # test scalar equality operation
    def test_scalar_eq(self):
        tmp = DualNumbers(1, -1)
        assert (tmp == z1) == (True, True)
        tmp = DualNumbers(1, 0)
        assert (tmp == z1) == (True, False)
        tmp = DualNumbers(2, -1)
        assert (tmp == z1) == (False, True)
        tmp = DualNumbers(2, -2)
        assert (tmp == z1) == (False, False)
        
    # test array equality operation
    def test_1d_eq(self):
        tmp = DualNumbers(1, np.array([0, 2]))
        assert (tmp == x1) == (True, True)
        tmp = DualNumbers(1, np.array([0, 1]))
        assert (tmp == x1) == (True, False)
        tmp = DualNumbers(0, np.array([0, 2]))
        assert (tmp == x1) == (False, True)
        tmp = DualNumbers(0, np.array([1, 3]))
        assert (tmp == x1) == (False, False)
    
    # test scalar inequality operation
    def test_scalar_ne(self):
        tmp = DualNumbers(2,-2)
        assert (tmp != z1) == (True, True)
        tmp = DualNumbers(2,-1)
        assert (tmp != z1) == (True, False)
        tmp = DualNumbers(1,-2)
        assert (tmp != z1) == (False, True)
        tmp = DualNumbers(1,-1)
        assert (tmp != z1) == (False, False)
    
    # test array inequality operation
    def test_1d_ne(self):
        tmp = DualNumbers(0, np.array([1, 3]))
        assert (tmp != x1) == (True, True)
        tmp = DualNumbers(0, np.array([0, 2]))
        assert (tmp != x1) == (True, False)
        tmp = DualNumbers(1, np.array([1, 3]))
        assert (tmp != x1) == (False, True)
        tmp = DualNumbers(1, np.array([0, 2]))
        assert (tmp != x1) == (False, False)
    