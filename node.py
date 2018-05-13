from abc import ABCMeta, abstractmethod
import numpy as np

class Node:
    '''
    Defines a single node in the graph. 'children' represent computations which
    must be done before the current node can be evaluated, and 'parents' are 
    nodes which depend on the current node.

    Each node has a value (a numpy array), which has a fixed shape. Attempting
    to set a value of a different shape will raise an exception.
    '''
    __metaclass__ = ABCMeta

    def __init__(self, children=None, value=None, shape=None):
        '''
        Initialize node's value and children list, and register node as a parent
        to each of its children. Either a value or a shape must be specified.
        If the value is not a numpy array, a numpy array is constructed from
        that value and used as the value of the node. If the value is a scalar
        of any kind (as np.isscalar), it is converted to a numpy array of shape
        (1,)
        '''
        if children:
            self.children = children
        else:
            self.children = []
        
        if value is None and shape is None:
            raise ValueError('Node must specify either a value or a shape')
        elif value is not None:
            if shape and shape != value.shape:
                raise ValueError('Requested shape {} is different than value shape {}'.format(shape, self.value.shape))
            
            # Convert any type of scalar to an array of shape (1,)
            if np.isscalar(value):
                if isinstance(value, np.ndarray):
                    value = np.array([np.asscalar(value)])
                else:
                    value = np.array([value])
            
            self.shape = value.shape
            self.value = value
        else:
            self.shape = shape
            self.value = np.zeros(shape=shape)


        self.parents = []
        # Register as parent to child nodes
        for child in self.children:
            child.parents.append(self)
    
    def __setattr__(self, name, value):
        if name == 'value':
            if value.shape != self.shape:
                raise ValueError('Wrong value shape {}; should be {}.'.format(self.shape, value.shape))
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

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
        '''
        Overloaded multiplication operator for nodes. If both operands have the
        same size, it defaults to element-wise multiplication. Otherwise, scalar
        multiplication or matrix-vector multiplication is chosen as appropriate.
        '''
        if not isinstance(other, Node):
            other = Constant(other)

        if self.shape == other.shape:
            return ElemMultiply(self, other)
        elif sum(self.shape) <= 1:
            return ScalarMultiply(self, other)
        elif sum(other.shape) <= 1:
            return ScalarMultiply(other, self)
        elif len(self.shape) == 2 and len(other.shape) == 1:
            if self.shape[1] != other.shape[0]:
                return NotImplemented
            else:
                return MatVecMultiply(self, other)
        else:
            return NotImplemented
    
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
        if left.shape != right.shape:
            raise ValueError('Element-wise addition only applies for children of the same shape.')
        super().__init__(children=[left, right], shape=left.shape)
    
    def calc_value(self):
        self.value = sum(c.value for c in self.children)
    
    def child_gradients(self, gradient):
        return [gradient, gradient]

class ElemMultiply(Node):
    '''
    A node which represents element-wise multiplication of its children
    '''
    def __init__(self, left, right):
        if left.shape != right.shape:
            raise ValueError('Element-wise multiplication only applies to children of the same shape.')
        super().__init__(children=[left, right], shape=left.shape)
    
    def calc_value(self):
        self.value = np.multiply(self.children[0].value, self.children[1].value)

    def child_gradients(self, gradient):
        return [gradient*self.children[1].value, gradient*self.children[0].value]

class ScalarMultiply(Node):
    '''
    A node which represents scalar multiplication. First child is scalar, and
    second is vector, matrix, etc.
    '''
    def __init__(self, scalar, vector):
        super().__init__(children=[scalar, vector], shape=vector.shape)
    
    def calc_value(self):
        self.value = np.multiply(self.children[0].value, self.children[1].value)

    def child_gradients(self, gradient):
        return [
            np.dot(gradient, self.children[1].value), 
            np.multiply(gradient, self.children[0].value*np.ones_like(self.children[1].value)),
        ]

class MatVecMultiply(Node):
    '''
    A node which represents multiplication of a matrix and a vector.
    '''
    def __init__(self, matrix, vector):
        super().__init__(children=[matrix, vector], shape=(matrix.shape[0],))

    def calc_value(self):
        self.value = np.dot(self.children[0].value, self.children[1].value)
    
    def child_gradients(self, gradient):
        return [
            np.outer(gradient, self.children[1].value),
            np.dot(gradient.T, self.children[0].value)
        ]

class Negate(Node):
    '''
    A node which negates its child node, element-wise.
    '''
    def __init__(self, child):
        super().__init__(children=[child], shape=child.shape)
    
    def calc_value(self):
        self.value = -self.children[0].value
    
    def child_gradients(self, gradient):
        return [-gradient]

class Functor:
    def __init__(self, function):
        self.function = (function,)
    
    def __call__(self, *args, **kwargs):
        return self.function[0](*args, **kwargs)
    

class ElemFunc(Node):
    '''
    A node which evaluates to an element-wise function applied to its child
    node value. It is initialized with the function (which is applied to the
    entire numpy array) and its derivative.
    '''
    def __init__(self, child, function, derivative):
        super().__init__(children=[child], shape=child.shape)
        self.function = Functor(function)
        self.derivative = Functor(derivative)
    
    def calc_value(self):
        self.value = self.function(self.children[0].value)
    
    def child_gradients(self, gradient):
        return [np.multiply(gradient, self.derivative(self.value))]