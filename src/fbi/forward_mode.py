# Authors: Wenxu Liu, Queenie Luo, Guangya Wan, Mengyao Zheng, Di Zhen                 #
# Course: AC207/CS107                                                                  #
# File: forward_mode.py                                                                 #
# Description: Perform forward mode automatic differentiation method, enabling a user  #
# to output just the function values, just the derivative values, or both the function #
# and derviative values in a tuple
########################################################################################


import numpy as np
from dual_number import DualNumbers


class ForwardMode:
    """
    A class to perform forward mode automatic differentiation mode, enabling a user
    to output just the function values evaluated at the evaluation point, just the derivative values, 
    or both the function and derviative values in a tuple.
    
    Instance Variables
    ----------
    input_values: a scalar or a vector which indicates the evaluation point, in this Milestone, only a scalar is allowed
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
    >>> fm.calculate_dual_number()
    (2, array([-2.]))
    
    """

    def __init__(self, input_values, input_function, seed = 1):
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
        -1

        """

        return self.calculate_dual_number()[1]
    
    def calculate_dual_number(self):
        """
        Parameters
        ----------
        None
        
        Returns
        -------
        evaluated value and derivative of the input function at the evaluation point
        
        Examples
        --------
        # get univariate scalar function value and jacobian
        >>> func = lambda x: x + 1
        >>> fm = forward_mode(1, func, -1)
        >>> fm.calculate_dual_number()
        (2, -1)
        
        """
        
        
        
        
        # handle the case of having only a single input value: convert the scalar value into a list
        
            
        # check if the input is a scalar
        if type(self.inputs)==float or type(self.inputs)==int:
            inputs_arr = np.array([self.inputs])
        elif len(self.inputs) == 1:
            inputs_arr = np.array([self.inputs])
        else:
            raise TypeError("ERROR: Input value is not a scaler")

        
        num_del = [0]
        num_del[0] = DualNumbers(inputs_arr[0], self.seed)
        
        z = self.functions(*num_del)
        
        try:
            return z.val, z.derv
        except AttributeError:
            print("ERROR: The input function must output a scalar.")
    
    