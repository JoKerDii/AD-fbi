import pytest
import numpy as np

from fbi.forward_mode import ForwardMode


##initialize ForwardMode objects

func1 = lambda x: x
fm1 = ForwardMode(1, func1)
fm2 = ForwardMode(1, func1, -1)

func2 = lambda x: x**2 + 2
fm3 = ForwardMode(2, func2)
fm4 = ForwardMode(3, func2, -2)



class TestForwardMode:
    """Test class for ForwardMode module"""
    
    # test attribute initialization
    def test_init(self):
        assert fm1.inputs == 1
        assert fm1.functions == func1
        assert fm1.seed == 1
        
        assert fm2.inputs == 1
        assert fm2.functions == func1
        assert fm2.seed == -1
        
        assert fm3.inputs == 2
        assert fm3.functions == func2
        assert fm3.seed == 1
        
        assert fm4.inputs == 3
        assert fm4.functions == func2
        assert fm4.seed == -2
        
        
    def test_get_fx_value(self):
        assert fm1.get_fx_value() == 1
        assert fm2.get_fx_value() == 1
        assert fm3.get_fx_value() == 6
        assert fm4.get_fx_value() == 11
        
    def test_get_derivative(self):
        assert fm1.get_derivative() == 1 
        assert fm2.get_derivative() == -1 
        assert fm3.get_derivative() == 4
        assert fm4.get_derivative() == -12
        
    def test_calculate_dual_number(self):
        assert fm1.calculate_dual_number() == (1, 1)
        assert fm2.calculate_dual_number() == (1, -1)
        assert fm3.calculate_dual_number() == (6, 4)
        assert fm4.calculate_dual_number() == (11, -12)
        
        
    
# func = lambda x: x + 1
# fm = ForwardMode(1, func, -1)
# print(fm.calculate_dual_number())
# print(fm.get_fx_value())
# print(fm.get_derivative())

        
# func = lambda x: x + 1
# fm = ForwardMode(1, func, -1)
# print(fm.calculate_dual_number() == (2, -1))
# print(fm.get_fx_value() == 2)
# print(fm.get_derivative() == -1)


# func2 = lambda x: (x, 2*x, x**2)
# fm3 = ForwardMode(1, func2, -1)
# fm3.calculate_dual_number()