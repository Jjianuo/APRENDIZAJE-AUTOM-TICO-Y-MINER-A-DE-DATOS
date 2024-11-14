import numpy as np
import copy
import math

from LinearRegressionMulti import LinearRegMulti

class LogisticRegMulti(LinearRegMulti):

    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model
        y (ndarray): Shape (m,) the real values of the prediction
        w, b (scalar): Parameters of the model
        lambda: Regularization parameter. Most be between 0..1. 
        Determinate the weight of the regularization.
    """
    def __init__(self, x, y,w,b, lambda_):
        super().__init__(x,y,w,b,lambda_)

    def f_w_b(self, x):
        aux = super().f_w_b(x)
        ret = 1 + np.exp(-(aux))
        ret = 1 / ret
        return ret
 
    def compute_cost(self):
        y_deriv = self.f_w_b(self.x)
        aux1 = self.y * np.log(y_deriv)
        aux2 = (1 - self.y)
        aux3 = np.log(1 - y_deriv)
        total_cost = -np.sum(aux1 + (aux2 * aux3))
        total_cost /= self.m
        return total_cost + self._regularizationL2Cost()

    
def cost_test_multi_obj(x,y,w_init,b_init):
    lr = LogisticRegMulti(x,y,w_init,b_init,0)
    cost = lr.compute_cost()
    return cost

def compute_gradient_multi_obj(x,y,w_init,b_init):
    lr = LogisticRegMulti(x,y,w_init,b_init,0)
    dw,db = lr.compute_gradient()
    return dw,db
