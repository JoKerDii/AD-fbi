# Authors: Wenxu Liu, Queenie Luo, Guangya Wan, Mengyao Zheng, Di Zhen          #
# Course: AC207/CS107                                                                #
# File: dual_number.py                                                          #
# Description: This class defines the dual number object to be used in forward mode 
# automatic differentiation. It contains methods to initialize the object, set and get    #
# the function and derivative value of the object, overload elementary          #
# operations, and define elementary functions.                                  #
#################################################################################

import numpy as np

def is_numeric(x):
    r"""Method to check whether input x contains numeric elements or not
    
    Parameters
    ----------
    x: An object to be checked if the values contained within it are either integers or floats
    
    Returns
    -------
    True if a scaler is a float/integer, or all of the elements within object x are either integers or floats. False otherwise.

    Examples
    --------
    >>> x = 6
    >>> print(is_numeric(x))
    True
    >>> x = 'e'
    >>> print(is_numeric(x))
    False
    >>> x = [1,2]
    >>> print(is_numeric(x))
    False
    >>> x = [1,'cs107']
    >>> print(is_numeric(x))
    False
    """
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
    
class DualNumbers:
    r"""A class representing a variable object to be used in automatic differentiation
    
    The class contains methods to initialize the object, set and get the function and 
    derivative value of the object, overload elementary operations, and define elementary 
    functions that are used in forward mode calculation
    
    Instance Variables
    ----------
    val: value of the DualNumbers object
    derv: derivative(s) of the DualNumbers object
    
    Returns
    -------
    A DualNumbers object that contains the value and derivative
    
    Examples
    --------
    >>> z1 = DualNumbers(1, -1)
    >>> z2 = DualNumbers(2.2, 0)
    >>> print(z_1 + z_2)
    Values: 3.2, Derivatives: -1
    """
    
    def __init__(self, val, derv_seed):
        r"""A constructor to create DualNumbers object with a value and a derivative
        
        Parameters
        ----------
        val: integer or float object that represents the value of DualNumbers object
        derv_seed: integer or float object that represents the seed value for the derivative of DualNumbers object 
        
        Returns
        -------
        None
        """
        self.val = val
        self.derv = derv_seed

    @property
    def val(self):
        r"""A method to retrieve the value attribute of DualNumbers object
        
        Parameters
        ----------
        None
        
        Returns
        -------
        val attribute of DualNumbers object
        
        Examples
        --------
        >>> z = DualNumbers(1, 2)
        >>> print(z.val)
        1
        """
        return self._val

    @val.setter
    def val(self, val):
        r"""A method to set the value attribute of DualNumbers object
        
        Parameters
        ----------
        val: float or integer object that represents the value of the DualNumbers object
        
        Returns
        -------
        None
        
        Raises
        ------
        TypeError
            If input is a non-integer or non-float value or a 1D numpy array of non-integer or non-float values
        
        Examples
        --------
        >>> z = DualNumbers(1, 2)
        >>> z.val = 2
        >>> print(z.val)
        2
       """
        if is_numeric(val):
            self._val = val
        else:
            raise TypeError('Error: Input value should be an int or float')

    @property
    def derv(self):
        r"""A method to retrieve the derivative attribute of DualNumbers object
        
        Parameters
        ----------
        None
        
        Returns
        -------
        derv attribute of DualNumbers object
        
        Examples
        --------
        >>> z = DualNumbers(1, 2)
        >>> print(z.derv)
        2
        """
        return self._derv

    @derv.setter
    def derv(self, derv):
        r"""A method to set the derivative attribute of DualNumbers object
        
        Parameters
        ----------
        derv: float/integer object or 1D array of float/integer objects that represents DualNumbers derivative
        
        Returns
        -------
        None
        
        Raises
        ------
        TypeError
            If input is a non-integer or non-float value or contains a 1D numpy array of non-integer or non-float values
        
        Examples
        --------
        >>> z = DualNumbers(1, 2)
        >>> z.derv = 1
        >>> print(z.derv)
        1
        """
        if is_numeric(derv):
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
        r"""A method to overload the string representation for DualNumbers object
        
        Parameters
        ----------
        None
        
        Returns
        -------
        A string representation of DualNumbers object
        
        Examples
        --------
        >>> print(DualNumbers(1, 2))
        Values: 1, Derivatives: 2
        """
        return f'Values: {self.val}, Derivatives: {self.derv}'
    
    def __add__(self, other):
        r"""A method to perform addition operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object or DualNumbers object
        
        Returns
        -------
        A DualNumbers object as the result of the addition operation
        
        Examples
        --------
        >>> z1 = DualNumbers(1, -1)
        >>> z2 = DualNumbers(1, 2)
        >>> print(z1 + z2)
        Values: 2, Derivatives: 1
        
        >>> x1 = DualNumbers(1, np.array([-1, 0]))
        >>> x2 = DualNumbers(1, np.array([0, 2]))
        >>> print(x1 + x2)
        Values: 2, Derivatives:[-1  2]
        """
        # perform addition if other is a dual number
        try:
            f = self.val + other.val
            f_prime = self.derv + other.derv
        # perform addition if other is a real number
        except AttributeError:
            f = self.val + other
            f_prime = self.derv
        return DualNumbers(f, f_prime)
    
    def __radd__(self, other):
        r"""A method to perform reverse addition operation on a DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object
        
        Returns
        -------
        A DualNumbers object as the result of reverse addition operation
        
        Examples
        --------
        >>> z1 = 2
        >>> z2 = DualNumbers(1, -1)
        >>> print(z1 + z2)
        Values: 3, Derivatives: -1

        >>> x1 = 2
        >>> x2 = DualNumbers(1, np.array([0, -1]))
        >>> print(x1 + x2)
        Values: 2, Derivatives: [0  -1]
        """
        return self + other
    
    def __sub__(self, other):
        r"""A method to perform subtraction operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: A float/integer object or DualNumbers object
        
        Returns
        -------
        A DualNumbers object as the result of the subtraction operation
        
        Examples
        --------
        >>> z1 = DualNumbers(1, -1)
        >>> z2 = DualNumbers(1, 2)
        >>> print(z1 - z2)
        Values: 0, Derivatives: -3

        >>> x = DualNumbers(1, np.array([-1, 0]))
        >>> y = DualNumbers(1, np.array([0, 2]))
        >>> print(x - y)
        Values: 0, Derivatives: [-1 -2]
        """
        # perform subtraction if other is a dual number
        try:
            f = self.val - other.val
            f_prime = self.derv - other.derv
        # perform subtraction if other is a real number
        except AttributeError:
            f = self.val - other
            f_prime = self.derv
        return DualNumbers(f, f_prime)
    
    def __rsub__(self, other):
        r"""A method to perform reverse subtraction operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object
        
        Returns
        -------
        A DualNumbers object as the result of the reverse subtraction operation
        
        Examples
        --------
        >>> z1 = 1
        >>> z2 = DualNumbers(1, -1)
        >>> print(z1 - z2)
        Values: 0, Derivatives: 1

        >>> x1 = 1
        >>> x2 = DualNumbers(1, np.array([0, 2]))
        >>> print(x1 - x2)
        Values: 0, Derivatives: [0  -2]
        """
        return other + (-self)
    
    def __mul__(self, other):
        r"""A method to perform multiplication operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object or DualNumbers object
        
        Returns
        -------
        A DualNumbers object as the result of the multiplication operation
        
        Examples
        --------
        >>> z1 = DualNumbers(1, -1)
        >>> z2 = DualNumbers(1, 2)
        >>> print(z1 * z2)
        Values: 1, Derivatives: 1

        >>> x1 = DualNumbers(1, np.array([-1, 0]))
        >>> x2 = DualNumbers(1, np.array([0, 2]))
        >>> print(x1 * x2)
        Values: 1, Derivatives: [-1  2]
        """
        # perform multiplication if other is a dual number
        try:
            f = self.val * other.val
            f_prime = self.val * other.derv + self.derv * other.val
        # perform multiplication if other is a real number
        except AttributeError:
            f = self.val * other
            f_prime = self.derv * other
        return DualNumbers(f, f_prime)
    
    def __rmul__(self, other):
        r"""A method to perform the reverse multiplication operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object
        
        Returns
        -------
        A DualNumbers object as the result of the reverse multiplication operation
        
        Examples
        --------
        >>> z1 = 1
        >>> z2 = DualNumbers(1, -1)
        >>> print(z1 * z2)
        Values: 1, Derivatives: -1

        >>> x1 = 1
        >>> x2 = DualNumbers(1, np.array([-1, 0]))
        >>> print(x1 * x2)
        Values: 1, Derivatives: [-1  0]
        """
        return self * other
    
    def __truediv__(self, other):
        r"""A method to perform division operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object or DualNumbers object
        
        Returns
        -------
        A DualNumbers object as the result of the division operation
        
        Raises
        ------
        ZeroDivisionError if denominator in division is zero
        
        Examples
        --------
        >>> z1 = DualNumbers(1, -1)
        >>> z2 = DualNumbers(1, 2)
        >>> print(z1 / z2)
        Values: 1.0, Derivatives: -3.0

        >>> x = DualNumbers(1, np.array([-1, 0]))
        >>> y = DualNumbers(1, np.array([0, 2]))
        >>> print(x / y)
        Values: 1.0, Derivatives: [-1. -2.]

        >>> z1 = DualNumbers(1, -1)
        >>> z2 = DualNumbers(0, 2)
        >>> print(z1 / z2)
        ZeroDivisionError: Error: Denominator in division should not be 0

        >>> z1 = DualNumbers(1, -1)
        >>> z2 = 0
        >>> print(z1 / z2)
        ZeroDivisionError: Error: Denominator in division should not be 0
        """
        # perform division if other is a dual number
        try:
            # avoid zero division
            if other.val == 0:
                raise ZeroDivisionError("Error: Denominator in division should not be 0")
            f = self.val / other.val
            f_prime = (self.derv * other.val - self.val * other.derv) / (other.val ** 2)
            return DualNumbers(f, f_prime)
        # perform division if other is a real number
        except AttributeError:
            # avoid zero division
            if other == 0:
                raise ZeroDivisionError("Error: Denominator in division should not be 0")
            f = self.val / other
            f_prime = self.derv / other
            return DualNumbers(f, f_prime)
        
    def __rtruediv__(self, other):
        r"""A method to perform the reverse division operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object
        
        Returns
        -------
        A DualNumbers object as the result of the reverse division operation
        
        Raises
        ------
        ZeroDivisionError if denominator in division is zero
        
        Examples
        --------
        >>> z1 = 1
        >>> z2 = DualNumbers(1, -1)
        >>> print(z1 / z2)
        Values: 1.0, Derivatives: 1.0

        >>> x1 = 1
        >>> x2 = DualNumbers(1, np.array([0, -1]))
        >>> print(x1 / x2)
        Values: 1.0, Derivatives: [0.  1.]

        >>> z1 = 1
        >>> z2 = DualNumbers(0, -1)
        >>> print(z1 / z2)
        ZeroDivisionError: Error: Denominator in division should not be 0
        """
        if self.val == 0:
            raise ZeroDivisionError("Error: Denominator in division should not be 0")
        f = other / self.val
        f_prime = (- other * self.derv) / (self.val ** 2)
        return DualNumbers(f, f_prime)

    def __pow__(self, other):
        r"""A method to perform power operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object or DualNumbers object
        
        Returns
        -------
        A DualNumbers object as the result of the power operation
        
        Raises
        ------
        ValueError
            If negative number is raised to a fraction power with an even denominator
            If the power is less than 1 and differentiation occurs at 0
        
        Examples
        --------
        >>> z1 = DualNumbers(1, -1)
        >>> z2 = DualNumbers(1, 2)
        >>> print(z1 ** z2)
        Values: 1, Derivatives: -1.0

        >>> x = DualNumbers(1, np.array([-1, 0]))
        >>> y = DualNumbers(1, np.array([0, 2]))
        >>> print(x ** y)
        Values: 1, Derivatives: [-1. 0.]

        >>> z1 = DualNumbers(-1, -1)
        >>> z2 = DualNumbers(0.2, 2)
        >>> print(z1 ** z2)
        ValueError: Error: Attempted to raise a negative number to a fraction power with even denominator

        >>> z1 = DualNumbers(0, -2)
        >>> z2 = DualNumbers(0.2, 2)
        >>> print(z1 ** z2)
        ValueError: Error: Attempted to find derivative at 0 when power is less than 1
        """
        # perform power operation if other is a dual number
        try:
            # avoid raising a negative number to a fraction power with an even denominator
            if self.val < 0 and other.val % 1 != 0 and other.val.as_integer_ratio()[1] % 2 == 0:
                raise ValueError("Error: Attempted to raise a negative number to a fraction power with even denominator")
            # avoid having a 0 derivative when the power is less than 1
            if self.val == 0 and other.val < 1:
                raise ValueError("Error: Attempted to find derivative at 0 when the power is less than 1")

            f = self.val ** other.val
            f_prime = (self.val ** (other.val - 1)) * self.derv * other.val + (
                    self.val ** other.val) * other.derv * np.log(self.val)
            return DualNumbers(f, f_prime)

        # perform power operation if other is a real number
        except AttributeError:
            # avoid raising a negative number to a fraction power with an even denominator
            if self.val < 0 and other % 1 != 0 and other.as_integer_ratio()[1] % 2 == 0:
                raise ValueError("Error: Attempted to raise a negative number to a fraction powerwith even denominator")
            # avoid having a 0 derivative when the power is less than 1
            if self.val == 0 and other < 1:
                raise ValueError("Error: Attempted to find derivative at 0 when power is less than 1")

            f = self.val ** other
            f_prime = other * self.val ** (other - 1)
            return DualNumbers(f, self.derv * f_prime)
        
    def __rpow__(self, other):
        r"""A method to perform the reverse power operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        other: float/integer object
        
        Returns
        -------
        A DualNumbers object as the result of the reverse power operation
        
        Examples
        --------
        >>> z1 = 1
        >>> z2 = DualNumbers(1, -1)
        >>> print(z1 ** z2)
        Values: 1, Derivatives: -0.0

        >>> x1 = 1
        >>> x2 = DualNumbers(1, np.array([0, -1]))
        >>> print(x1 ** x2)
        Values: 1, Derivatives: [0.  -0.]
        """
        f = other ** self.val
        f_prime = (other ** self.val) * self.derv * np.log(other)
        return DualNumbers(f, f_prime)
    
    def __neg__(self):
        r"""A method to perform the negation operation on the DualNumbers object and the other object
        
        Parameters
        ----------
        None
        
        Returns
        -------
        A DualNumbers object as the result of the negation operation
        
        Examples
        --------
        >>> z = DualNumbers(1, -1)
        >>> print(-z)
        Values: -1, Derivatives: 1

        >>> x = DualNumbers(1, np.array([-1, 0]))
        >>> print(-x)
        Values: -1, Derivatives: [1  0]
        """
        return DualNumbers(-1 * self.val, -1 * self.derv)
    
    def __eq__(self, other):
        r"""A method to check whether the DualNumbers objects are equal to each other
        
        Parameters
        ----------
        other: A DualNumbers object
        
        Returns
        -------
        A tuple of boolean variables where the first variable indicates whether the function values are equal and the second
        variable indicates if the derivative values are equal
        
        Examples
        --------
        >>> z1 = DualNumbers(1, 10)
        >>> z2 = DualNumbers(1, 10)
        >>> print(z1 == z2)
        (True, True)

        >>> z1 = DualNumbers(1, 10)
        >>> z2 = DualNumbers(2, 10)
        >>> print(z1 == z2)
        (False, True)

        >>> z1 = DualNumbers(2, 1)
        >>> z2 = DualNumbers(1, 2)
        >>> print(z1 == z2)
        (False, False)
        """
        # check if DualNumbers values are equal
        try:
            is_val_eq = all(self.val == other.val)
        except TypeError:
            is_val_eq = True if self.val == other.val else False

        # check if DualNumbers derivatives are equal
        try:
            is_derv_eq = all(self.derv == other.derv)
        except TypeError:
            is_derv_eq = True if self.derv == other.derv else False
        return is_val_eq, is_derv_eq
    
    def __ne__(self, other):
        r"""A method to check whether the DualNumbers objects are not equal to each other
        
        Parameters
        ----------
        other: A DualNumbers object
        
        Returns
        -------
        A tuple of boolean variables where the first variable indicates if the function values are not equal and
        the second variable indicates if the derivative values are not equal
        
        Examples
        --------
        >>> z1 = DualNumbers(1, 10)
        >>> z2 = DualNumbers(1, 10)
        >>> print(z1 != z2)
        (False, False)

        >>> z1 = DualNumbers(1, 10)
        >>> z2 = DualNumbers(2, 10)
        >>> print(z1 != z2)
        (True, False)

        >>> z1 = DualNumbers(2, 1)
        >>> z2 = DualNumbers(1, 2)
        >>> print(z1 != z2)
        (True, True)
        """
        # check if DualNumbers values are not equal
        try:
            is_val_eq = all(self.val != other.val)
        except TypeError:
            is_val_eq = True if self.val != other.val else False

        # check if DualNumbers derivatives are not equal
        try:
            is_derv_eq = all(self.derv != other.derv)
        except TypeError:
            is_derv_eq = True if self.derv != other.derv else False

        return is_val_eq, is_derv_eq