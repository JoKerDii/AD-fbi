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
            raise TypeError('ERROR: Input value should be an int or float')

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
                raise ValueError('ERROR: Input value should be an int or float')
            self._derv = derv
        # for all other non-numeric cases, raise the appropriate value error
        else:
            raise TypeError('ERROR: Input value must be an array of ints/floats or be a scalar int/float')

    def __repr__(self):
        return f'Values:{self.val}, Derivatives:{self.derv}'