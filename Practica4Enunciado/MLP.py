import numpy as np

class MLP:

    """
    Constructor: Computes MLP.

    Args:
        theta1 (array_like): Weights for the first layer in the neural network.
        theta2 (array_like): Weights for the second layer in the neural network.
    """
    def __init__(self,theta1,theta2):
        self.theta1 = theta1
        self.theta2 = theta2
        
    """
    Num elements in the training data. (private)

    Args:
        x (array_like): input data. 
    """
    def _size(self,x):
        return x.shape[0]
    
    """
    Computes de sigmoid function of z (private)

    Args:
        z (array_like): activation signal received by the layer.
    """
    def _sigmoid(self,z):
        return 0

    """
    Run the feedwordward neural network step

    Args:
        z (array_like): activation signal received by the layer.

	Return 
	------
	a1,a2,a3 (array_like): activation functions of each layers
    z2,z3 (array_like): signal fuction of two last layers
    """
    def feedforward(self,x):
        a1,a2,a3,z2,z3 = 0
        return a1,a2,a3,z2,z3 # devolvemos a parte de las activaciones, los valores sin ejecutar la función de activación


    """
    Computes only the cost of a previously generated output (private)

    Args:
        yPrime (array_like): output generated by neural network.
        y (array_like): output from the dataset

	Return 
	------
	J (scalar): the cost.
    """
    def compute_cost(self, yPrime,y): # calcula solo el coste, para no ejecutar nuevamente el feedforward.
        J = 0
        return J
    

    """
    Get the class with highest activation value

    Args:
        a3 (array_like): output generated by neural network.

	Return 
	------
	p (scalar): the class index with the highest activation value.
    """
    def predict(self,a3):
        p = -1
        return p

    