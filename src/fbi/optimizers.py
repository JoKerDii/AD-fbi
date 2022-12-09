# Authors: Wenxu Liu, Queenie Luo, Guangya Wan, Mengyao Zheng, Di Zhen          #
# Course: AC207/CS107                                                           #
# File: optimzers.py                                                            #
# Description: Create three different optimization techniques that leverage     #
# forward mode automatic differentiation, enabling a user to find the minimum   #
# value of a function, location of the minimum value, and wall clock time to    #
# find these values.                                                            #
#################################################################################

import numpy as np
from forward_mode import ForwardMode
import time

class Optimizer:
    """
    A class containing three different optimizer methods that leverage forward mode
    automatic differentiation, allowing a user to to find the minimum value of a function,
    location of the minimum value, and wall clock time required to find these values

    Examples
    --------
    # sample use case of instantiating and using a momentum optimizer
    >>> x = 2
    >>> fx = lambda x: x**4
    >>> print(Optimizer.momentum(x, fx, 1000))
    (0.04814004898071289, 2.0278841795288425e-05, array([-0.06710591]))
    """
    @staticmethod
    def momentum(x, fx, num_iter = 10000, alpha=0.01, beta=.9, verbose = False):
        """
        Parameters
        ----------
        x: the variable input (can be in either scalar or vector form)
        fx: the function you would like to obtain the minimum for
        num_iter: the number of interations to perform (default 10,000)
        alpha: learning rate for the gradiant descent (default 0.01)
        beta: exponential decay (default 0.9)

        Returns
        -------
        opt_time: The time it takes to run the optimizer in seconds
        val: the position of the minimum value
        curr_val: the minimum value (can be in either scalar or vector form)

        Examples
        --------
        >>> x = 1
        >>> fx = lambda x: (-1 * x.log()) + (x.exp() * x**4) / 10
        >>> Optimizer.momentum(x, fx, 1000)
        (0.1293182373046875, 0.26172998379097046, array([0.94233316]))

        >>> x = np.array([1, -1])
        >>> fx = lambda x, y:x**3 + y**2
        >>> Optimizer.momentum(x, fx, 1000)
        (0.08369302749633789, 2.7605629377339922e-05, array([ 3.02226506e-02, -3.30135704e-12]))

        >>> x = 2
        >>> fx = lambda x: (x - 1)**2 + 5
        >>> Optimizer.momentum(x, fx, 1000)
        (0.06605792045593262, 5.0, array([1.]))
        """
        vals=[]
        currvals=[]
        # start the timer
        start = time.time()
        # decay value must be great than or equal to 0 and less than 1
        if 0 <= beta < 1 and 0 < alpha < 1:
            mt, curr_val = 0, x
            fm = ForwardMode(x, fx)
            val, x_der = fm.get_fx_value(), fm.get_derivative()
            # perform momentum optimization for the number of iterations specified
            for t in range(1, num_iter + 1):
                # calculate momentum
                mt = beta * mt + (1 - beta) * x_der
                # compute the new variation to update the current x location
                variation = alpha * mt
                curr_val = curr_val - variation
                # recalculate the function value and derivative at the updated value
                fm = ForwardMode(curr_val, fx)
                val, x_der = fm.get_fx_value(), fm.get_derivative()
                # store val and curr_val
                if t % 10 == 0:
                    vals.append(val)
                    currvals.append(curr_val)
        # raise the appropriate error for beta value not between 0 to 1
        elif beta>=1 or beta < 0:
            raise ValueError("Beta Values must be within the range of [0,1)")
        # raise the appropriate error for alpha value not between 0 to 1
        elif alpha >=1 or alpha <= 0:
            raise ValueError("Learning rate alpha must be within the range of (0,1)")
        # end the timer and compute wall clock time
        end = time.time()
        opt_time = end - start
        
        if verbose == False:
            return opt_time, val, curr_val
        else:
            return opt_time, vals, currvals
    
    
    
    @staticmethod
    def gradient_descent(x, fx, num_iter = 10000, alpha=0.001, verbose = False):
        """
        Parameters
        ----------
        x: the starting point to find the minimum
        fx: the function you would like to obtain the minimum for
        num_iter: the number of interations to perform (default 10,000)
        alpha: learning rate for the gradiant descent (default 0.001)
        

        Returns
        -------
        opt_time: The time it takes to run the optimizer in seconds
        val: the position of the minimum value
        curr_val: the minimum value (can be in either scalar or vector form)

        Examples
        --------
        >>> x = 1
        >>> fx = lambda x: (-1 * x.log()) + (x.exp() * x**4) / 10
        >>> Optimizer.gradient_descent(x, fx, 1000)
        (0.07654619216918945, 0.2617299837909705, array([0.94233316]))
        

        >>> x = np.array([1, -1])
        >>> fx = lambda x, y:x**3 + y**2
        >>> Optimizer.gradient_descent(x, fx, 1000)
        (0.05128312110900879,
        3.323299732460115e-05,
        array([ 3.21506559e-02, -1.68296736e-09]))
        

        >>> x = 2
        >>> fx = lambda x: (x - 1)**2 + 5
        >>> Optimizer.gradient_descent(x, fx, 1000)
        (0.036324501037597656, 5.0, array([1.]))
        """
        # initiate the array to store the function values
        vals=[]
        # initiate the array to store the intermediate values for the input variable(s)
        currvals=[]
        # start the timer
        start = time.time()
        # learning rate value must be great than or equal to 0 and less than 1
        if 0 < alpha < 1:
            curr_val = x
            fm = ForwardMode(x, fx)
            val, x_der = fm.get_fx_value(), fm.get_derivative()
            # perform gradient descent for the number of iterations specified
            for t in range(1, num_iter + 1):
                # compute the new variation to update the current x location
                variation = alpha * x_der
                
                curr_val = curr_val - variation
                # recalculate the function value and derivative at the updated value
                fm = ForwardMode(curr_val, fx)
                val, x_der = fm.get_fx_value(), fm.get_derivative()
                # store val and curr_val
                if t % 10 == 0:
                    vals.append(val)
                    currvals.append(curr_val)
        # raise the appropriate error for alpha value not between 0 to 1
        else:
            raise ValueError("Learning rate alpha must be within the range of (0,1)")
        # end the timer and compute wall clock time
        end = time.time()
        opt_time = end - start
        
        if verbose == False:
            return opt_time, val, curr_val
        else:
            return opt_time, vals, currvals
    

