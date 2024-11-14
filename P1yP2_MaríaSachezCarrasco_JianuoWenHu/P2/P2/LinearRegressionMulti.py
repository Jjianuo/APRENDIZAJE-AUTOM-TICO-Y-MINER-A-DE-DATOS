import numpy as np
import copy
import math
from VectorialRegression import VectorialReg

class LinearRegMulti(VectorialReg):

    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model
        y (ndarray): Shape (m,) the real values of the prediction
        w, b (scalar): Parameters of the model
        lambda: Regularization parameter. Most be between 0..1. 
        Determinate the weight of the regularization.
    """
    def __init__(self, x, y, w, b, lambda_):
        self.x = x
        self.y = y
        self.w = w
        self.b = b        
        self.lambda_ = lambda_

    
    def f_w_b(self, x):
        return np.dot(x, self.w) + self.b

    """
    Compute the regularization cost (is private method: start with _ )
    This method will be reuse in the future.

    Returns
        _regularizationL2Cost (float): the regularization value of the current model
    """
    
    def _regularizationL2Cost(self):
        w = self.w ** 2
        reg_cost = np.sum(w)
        reg_cost_final = (self.lambda_ / (2 * self.x.size)) * reg_cost
        return reg_cost_final
    
    """
    Compute the regularization gradient (is private method: start with _ )
    This method will be reuse in the future.

    Returns
        _regularizationL2Gradient (vector size n): the regularization gradient of the current model
    """ 
    
    def _regularizationL2Gradient(self):
        reg_gradient_final = (self.lambda_ / self.x.size) * self.w
        return reg_gradient_final

    def compute_cost(self):
        return super().compute_cost() + self._regularizationL2Cost()

    
    def compute_gradient(self):
        dj_dw, dj_db = super().compute_gradient()
        return dj_dw + self._regularizationL2Gradient(), dj_db

    
def cost_test_multi_obj(x,y,w_init,b_init):
    lr = LinearRegMulti(x,y,w_init,b_init,0)
    cost = lr.compute_cost()
    return cost

def compute_gradient_multi_obj(x,y,w_init,b_init):
    lr = LinearRegMulti(x,y,w_init,b_init,0)
    dw,db = lr.compute_gradient()
    return dw,db
