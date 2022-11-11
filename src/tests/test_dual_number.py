import pytest

# import names to test
from fbi.dual_number import dual_number, var_type

z1 = dual_number(1,2)
class TestDualNumber:
    """Test class for dual_number module"""

    def test_init(self):
        assert z1.val == 1
        assert z1.derv == 2
        
    def test_setter(self):
        z1.val = 2
        z1.derv = 1
        assert z1.val == 2
        assert z1.derv == 1
