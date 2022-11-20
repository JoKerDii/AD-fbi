# Authors: Wenxu Liu, Queenie Luo, Guangya Wan, Mengyao Zheng, Di Zhen                 #
# Course: AC207/CS107                                                                  #
# File: forward_mode.py                                                                 #
# Description: Perform forward mode automatic differentiation method, enabling a user  #
# to output just the function values, just the derivative values, or both the function #
# and derviative values in a tuple
########################################################################################


import numpy as np
from fbi.dual_number import DualNumbers, is_numeric


class ForwardMode:
    """
    A class to perform forward mode automatic differentiation mode, enabling a user
    to output just the function values evaluated at the evaluation point, just the derivative values, 
    or both the function and derviative values in a tuple.
    
    Instance Variables
    ----------
    input_values: a scalar or a vector which indicates the evaluation point
    input_function: a scalar function or a vector of functions 
    seed: a seed vector (optional parameter: default value = 1)
    
    Examples
    --------
    # get function value
    >>> func = lambda x: x**2 + 1
    >>> fm = ForwardMode(1, func, -1)
    >>> fm.get_fx_value()
    2
    # get function derivative
    >>> fm.get_derivative()
    array([-2.])
    # get function value and derivative
    >>> fm.get_function_value_and_jacobian()
    (2, array([-2.]))

    
    
    
    """

    def __init__(self, input_values, input_function, seed=1):
        self.inputs = input_values
        self.functions = input_function
        self.seed = seed

    def get_fx_value(self):
        """
        Parameters
        ----------
        None
        
        Returns
        -------
        evaluated value of the input function
        
        Examples
        --------
        # get univariate scalar function value
        >>> func = lambda x: x
        >>> fm = ForwardMode(1, func, -1)
        >>> fm.get_fx_value()
        1
        # get multivariate scalar function value
        >>> func = lambda x, y: x + y
        >>> fm = ForwardMode(np.array([1, 1]), func, [1, -1])
        >>> fm.get_fx_value()
        2
        """

        return self.calculate_dual_number()[0]
    

    def get_derivative(self):
        """
        Parameters
        ----------
        None
        
        Returns
        -------
        the derivative of the input function at the evaluation point
        
        Examples
        --------
        # get univariate scalar function derivative
        >>> func = lambda x: x
        >>> fm = forward_mode(1, func, -1)
        >>> fm.get_derivative()
        array([-1.])
        # get multivariate scalar function jacobian
        >>> func = lambda x, y: x + y
        >>> fm = forward_mode(np.array([1, 1]), func, [1, -1])
        >>> fm.get_derivative()
        array([ 1., -1.])
        """

        return self.calculate_dual_number()[1]
    
    def calculate_dual_number(self):
        self.inputs = np.array([self.inputs])

        num_inp = len(self.inputs)
        seed_vec = np.zeros(num_inp)

        num_del = [0] * num_inp
        for i in range(num_inp):
            num_del[i] = DualNumbers(self.inputs[i], self.seed)
        
        z = self.functions(*num_del)
        return z.val, z.derv
    