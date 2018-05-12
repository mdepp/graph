import numpy as np
import node

def relu_function(x):
    '''
    Rectifier activation function. Expects 1-dimension numpy array as input.
    '''
    return np.max(np.array([x, np.zeros_like(x)]), axis=0)

def relu_derivative(x):
    '''
    Derivative of rectifier activation function.
    '''
    return np.heaviside(x, 0)


def relu(child_node):
    '''
    Given a node, returns an ElemFunc node which applies the relu activation
    to the given node.
    '''
    return node.ElemFunc(child_node, relu_function, relu_derivative)