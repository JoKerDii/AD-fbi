# Authors: Wenxu Liu, Queenie Luo, Guangya Wan, Mengyao Zheng, Di Zhen          #
# Course: AC 207                                                                #
# File: dual_number.py                                                          #
# Description: This class defines a variable object to be used in automatic     #
# differentiation. It contains methods to initialize the object, set and get    #
# the function and derivative value of the object, overload elementary          #
# operations, and define elementary functions.                                  #
#################################################################################

import numpy as np

def var_type(x):
    # if the input object is not a character but a scalar, it must be numeric
    if not isinstance(x, str) and np.isscalar(x):
        return True
    # if the input object is an array, check all contents
    else:
        # if an element is numeric, move on to the next, otherwise it's not valid
        for i in x:
            if isinstance(i, (int, np.int32, np.int64, float, np.float64)):
                pass
            else:
                return False
        return True
    
class dual_number:
    
    def __init__(self, val, derv_seed):
        self.val = val
        self.derv = derv_seed

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        if var_type(val):
            self._val = val
        else:
            raise TypeError('Error: Input value should be an int or float')

    @property
    def derv(self):
        return self._derv

    @derv.setter
    def derv(self, derv):
        if var_type(derv):
            self._derv = derv
        # in the case of a 1D array of derivatives, check each element individually
        elif isinstance(derv, np.ndarray) and len(derv.shape) == 1:
            try:
                derv = derv.astype(float)
            except ValueError:
                raise ValueError('Error: Input value should be an int or float')
            self._derv = derv
        # for all other non-numeric cases, raise the appropriate value error
        else:
            raise TypeError('Error: Input value must be an array of ints/floats or be a scalar int/float')

    def __repr__(self):
        return f'Values: {self.val}, Derivatives: {self.derv}'
    
    def __add__(self, other):
        # perform addition if other is a dual number
        try:
            f = self.val + other.val
            f_prime = self.derv + other.derv
        # perform addition if other is a real number
        except AttributeError:
            f = self.val + other
            f_prime = self.derv
        return dual_number(f, f_prime)
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        # perform subtraction if other is a dual number
        try:
            f = self.val - other.val
            f_prime = self.derv - other.derv
        # perform subtraction if other is a real number
        except AttributeError:
            f = self.val - other
            f_prime = self.derv
        return dual_number(f, f_prime)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __mul__(self, other):
        # perform multiplication if other is a dual number
        try:
            f = self.val * other.val
            f_prime = self.val * other.derv + self.derv * other.val
        # perform multiplication if other is a real number
        except AttributeError:
            f = self.val * other
            f_prime = self.derv * other
        return dual_number(f, f_prime)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        # perform division if other is a dual number
        try:
            # avoid zero division
            if other.val == 0:
                raise ZeroDivisionError("Error: Denominator in division should not be 0")
            f = self.val / other.val
            f_prime = (self.derv * other.val - self.val * other.derv) / (other.val ** 2)
            return dual_number(f, f_prime)
        # perform division if other is a real number
        except AttributeError:
            # avoid zero division
            if other == 0:
                raise ZeroDivisionError("Error: Denominator in division should not be 0")
            f = self.val / other
            f_prime = self.derv / other
            return dual_number(f, f_prime)
        
    def __rtruediv__(self, other):
        if self.val == 0:
            raise ZeroDivisionError("Error: Denominator in division should not be 0")
        f = other / self.val
        f_prime = (- other * self.derv) / (self.val ** 2)
        return dual_number(f, f_prime)

    def __pow__(self, other):
        # perform power operation if other is a dual number
        try:
            # ensure user does not raise a negative number to a fraction with an even denominator
            if self.val < 0 and other.val % 1 != 0 and other.val.as_integer_ratio()[1] % 2 == 0:
                raise ValueError("Error: Attempted to raise a negative number to a fraction with even denominator")
            # ensure user does not have a 0 derivative when the exponent is less than 1
            if self.val == 0 and other.val < 1:
                raise ValueError("Error: Attempted to find derivative at 0 when exponent is less than 1")

            f = self.val ** other.val
            # compute the derivative power rule for a dual number exponent
            f_prime = (self.val ** (other.val - 1)) * self.derv * other.val + (
                    self.val ** other.val) * other.derv * np.log(self.val)
            return dual_number(f, f_prime)

        # perform power operation if other is a real number
        except AttributeError:
            # ensure user does not raise a negative number to a fraction with an even denominator
            if self.val < 0 and other % 1 != 0 and other.as_integer_ratio()[1] % 2 == 0:
                raise ValueError("ERROR: Attempted to raise a negative number to a fraction with even denominator")
            # ensure user does not have a 0 derivative when the exponent is less than 1
            if self.val == 0 and other < 1:
                raise ValueError("ERROR: Attempted to find derivative at 0 when exponent is less than 1")

            f = self.val ** other
            # compute the derivative power rule for a real number exponent
            f_prime = other * self.val ** (other - 1)
            return dual_number(f, self.derv * f_prime)
        
    def __rpow__(self, other):
        f = other ** self.val
        f_prime = (other ** self.val) * self.derv * np.log(other)
        return dual_number(f, f_prime)
    
    def __neg__(self):
        return dual_number(-1 * self.val, -1 * self.derv)
    
    def __eq__(self, other):
        # check if val_derv values are equal
        try:
            value_eq = all(self.val == other.val)
        except TypeError:
            value_eq = True if self.val == other.val else False

        # check if val_derv derivatives are equal
        try:
            derivative_eq = all(self.derv == other.derv)
        except TypeError:
            derivative_eq = True if self.derv == other.derv else False
        return value_eq, derivative_eq
    
    def __ne__(self, other):
        # check if val_derv values are not equal
        try:
            value_eq = all(self.val != other.val)
        except TypeError:
            value_eq = True if self.val != other.val else False

        # check if val_derv derivatives are not equal
        try:
            derivative_eq = all(self.derv != other.derv)
        except TypeError:
            derivative_eq = True if self.derv != other.derv else False

        return value_eq, derivative_eq