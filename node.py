from abc import ABCMeta, abstractmethod
import numpy as np

class Node:
    '''
    Defines a single node in the graph. 'children' represent computations which
    must be done before the current node can be evaluated, and 'parents' are 
    nodes which depend on the current node.
    '''
    __metaclass__ = ABCMeta

    def __init__(self):
        self.children = []
        self.parents = []
        self.value = None
    
    @abstractmethod
    def calc_value(self):
        '''
        Set the node's value from the pre-calculated values of all child nodes.
        '''
        pass

    @abstractmethod
    def child_gradients(self, gradient):
        '''
        Calculate gradients of each child node when given the gradient of this
        node. Returns a list of gradients, each corresponding to one child node.
        '''
        pass


    def __add__(self, other):
        if not isinstance(other, Node):
            other = Constant(other)
        
        res = Add(self, other)
        self.parents.append(res)
        other.parents.append(res)
        return res
    
    __radd__ = __add__ # Since addition is commutative

    def __mul__(self, other):
        if not isinstance(other, Node):
            other = Constant(other)
        
        res = Multiply(self, other)
        self.parents.append(res)
        other.parents.append(res)
        return res
    
    __rmul__ = __mul__

    def __sub__(self, other):
        if not isinstance(other, Node):
            other = Constant(other)
        
        negated = Negate(other)
        other.parents.append(negated)
        res = Add(self, negated)
        self.parents.append(res)
        negated.parents.append(res)
        return res

class Variable(Node):
    '''
    A leaf which evaluates to data given as input
    '''
    def __init__(self, value=None):
        super().__init__()
        self.value = value
    
    def substitute(self, value):
        '''
        Set the value of this variable.
        '''
        self.value = value
    
    def calc_value(self):
        pass
    
    def child_gradients(self, gradient):
        return []


class Constant(Node):
    '''
    A leaf which represents a constant value
    '''
    def __init__(self, value):
        super().__init__()
        self.value = value
    
    def calc_value(self):
        pass
    
    def child_gradients(self, gradient):
        return []

class Add(Node):
    '''
    A node which evaluates to the sum of its children
    '''
    def __init__(self, left, right):
        super().__init__()
        self.children = [left, right]
    
    def calc_value(self):
        self.value = sum(c.value for c in self.children)
    
    def child_gradients(self, gradient):
        return [gradient, gradient]

class Multiply(Node):
    '''
    A node which represents multiplication of its children
    '''
    def __init__(self, left, right):
        super().__init__()
        self.children = [left, right]
    
    def calc_value(self):
        self.value = np.multiply(self.children[0].value, self.children[1].value)

    def child_gradients(self, gradient):
        return [gradient*self.children[1].value, gradient*self.children[0].value]

class Negate(Node):
    '''
    A node which negates values.
    '''
    def __init__(self, child):
        super().__init__()
        self.children = [child]
    
    def calc_value(self):
        self.value = -self.children[0].value
    
    def child_gradients(self, gradients):
        return [-gradients]