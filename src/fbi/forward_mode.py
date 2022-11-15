# Authors: Wenxu Liu, Queenie Luo, Guangya Wan, Mengyao Zheng, Di Zhen                 #
# Course: AC207/CS107                                                                  #
# File: forward_mode.py                                                                 #
# Description: Perform forward mode automatic differentiation method, enabling a user  #
# to output just the function values, just the derivative values, or both the function #
# and derviative values in a tuple
########################################################################################


import numpy as np
from fbi.dual_number import dual_number




class forward_mode:
    """
    A class to perform forward mode automatic differentiation mode, enabling a user
    to output just the function values evaluated at the evaluation point , just the derivative values, or both the
    function and derviative values in a tuple

    

    """

    def __init__(self, input_values, input_function, seed=1):
        self.inputs = input_values
        self.functions = input_function
        self.seed = seed

    def get_fx_value(self):
       

        return self.get_function_value_and_jacobian()[0]

    def get_jacobian(self):
        

        return self.get_function_value_and_jacobian()[1]

    