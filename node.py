from abc import ABCMeta, abstractmethod
import numpy as np

class Node:
    '''
    Defines a single node in the graph. 'children' represent computations which
    must be done before the current node can be evaluated, and 'parents' are 
    nodes which depend on the current node.
    '''
    __metaclass__ = ABCMeta

    def __init__(self, children=None, value=None):
        '''
        Initialize node's value and children list, and register node as a parent
        to each of its children.
        '''
        if children:
            self.children = children
        else:
            self.children = []
        self.parents = []
        self.value = value
        # Register as parent to child nodes
        for child in self.children:
            child.parents.append(self)
    
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
        
        return ElemAdd(self, other)
    
    __radd__ = __add__ # Since addition is commutative

    def __mul__(self, other):
        if not isinstance(other, Node):
            other = Constant(other)

        return ElemMultiply(self, other)
    
    __rmul__ = __mul__

    def __sub__(self, other):
        if not isinstance(other, Node):
            other = Constant(other)
        
        return ElemAdd(self, Negate(other))

class Variable(Node):
    '''
    A leaf which evaluates to data given as input
    '''
    def __init__(self, value=None):
        super().__init__(value=value)
    
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
        super().__init__(value=value)
    
    def calc_value(self):
        pass
    
    def child_gradients(self, gradient):
        return []

class ElemAdd(Node):
    '''
    A node which evaluates to the element-wise sum of its children
    '''
    def __init__(self, left, right):
        super().__init__(children=[left, right])
    
    def calc_value(self):
        self.value = sum(c.value for c in self.children)
    
    def child_gradients(self, gradient):
        return [gradient, gradient]

class ElemMultiply(Node):
    '''
    A node which represents element-wise multiplication of its children
    '''
    def __init__(self, left, right):
        super().__init__(children=[left, right])
    
    def calc_value(self):
        self.value = np.multiply(self.children[0].value, self.children[1].value)

    def child_gradients(self, gradient):
        return [gradient*self.children[1].value, gradient*self.children[0].value]

class Negate(Node):
    '''
    A node which negates its child node, element-wise.
    '''
    def __init__(self, child):
        super().__init__(children=[child])
    
    def calc_value(self):
        self.value = -self.children[0].value
    
    def child_gradients(self, gradient):
        return [-gradient]

class ElemFunc(Node):
    '''
    A node which evaluates to an element-wise function applied to its child
    node value. It is initialized with the function (which is applied to the
    entire numpy array) and its derivative.
    '''
    def __init__(self, child, function, derivative):
        super().__init__(children=[child])
        self.function = function
        self.derivative = derivative
    
    def calc_value(self):
        self.value = self.function(self.children[0].value)
    
    def child_gradients(self, gradient):
        return [np.multiply(gradient, self.derivative(self.value))]