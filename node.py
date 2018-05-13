from abc import ABCMeta, abstractmethod
import numpy as np

class Node:
    '''
    Defines a single node in the graph. 'children' represent computations which
    must be done before the current node can be evaluated, and 'parents' are 
    nodes which depend on the current node.

    Each node has a value (a numpy array), which has a fixed shape. Attempting
    to set a value of a different shape will raise an exception.

    The first dimension of the value is the batch dimension. This dimension can
    have size 1; if so, it is broadcasted as appropriate. The 'shape' of the
    node is the shape of value without the batch dimension.
    '''
    __metaclass__ = ABCMeta

    def __init__(self, children, shape, value=None):
        '''
        Initialize node's value and children list, and register node as a parent
        to each of its children. Shape must be specified, and is the shape of
        the node output, *without* the batch dimension. Shape is always extended
        to contain at least one dimension. If 'value' is specified, it is
        expected to be a numpy array of shape (?, shape) where ? is the
        batch dimension, unless it is a scalar, in which case the value is set
        to an array of shape (1, shape) with all elements equal to the scalar.
        '''
        if children:
            self.children = children
        else:
            self.children = []
        
        self.shape = shape # shape if len(shape) >= 1 else (1,)
        if value is not None:
            if np.isscalar(value):
                self.value = np.full((1, *self.shape), np.asscalar(value))
            elif self.shape != value.shape[1:]:
                raise ValueError('Value and shape are incompatible; value.shape = {} != {} = (None, shape)'.format(value.shape, (None, *self.shape)))
            else:
                self.value = value
        else:
            self.value = np.zeros((1, *self.shape))

        self.parents = []
        # Register as parent to child nodes
        for child in self.children:
            child.parents.append(self)
    
    def __setattr__(self, name, value):
        if name == 'value':
            if value.shape[1:] != self.shape:
                raise ValueError('Wrong value shape {}; should be {}).'.format(value.shape, (None, *self.shape)))
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

        If an argument is not a node, a Variable node is constructed for it,
        with constant value across all batches (it is assumed that the argument
        has no batch dimension).
        '''
        if not isinstance(other, Node):
            other = Variable(np.array([other]))

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
    A leaf which has a single value.
    '''
    def __init__(self, value):
        '''
        Initialize with the given value. This value must contain the batch
        dimension.
        '''
        super().__init__(children=[], shape=value.shape[1:], value=value)
    
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
        return [np.multiply(gradient, self.children[1].value),
                np.multiply(gradient, self.children[0].value)]

class ScalarMultiply(Node):
    '''
    A node which represents scalar multiplication. First child is scalar, and
    second is vector, matrix, etc.
    '''
    def __init__(self, scalar, vector):
        if len(scalar.shape) > 1:
            raise TypeError('{} is not a scalar.'.format(scalar))
        super().__init__(children=[scalar, vector], shape=vector.shape)
    
    def calc_value(self):
        self.value = np.multiply(self.children[0].value, self.children[1].value)

    def child_gradients(self, gradient):
        return [
            np.einsum('...j,...j->...', gradient, self.children[1].value),
            np.multiply(gradient, self.children[0].value),
        ]

class MatVecMultiply(Node):
    '''
    A node which represents multiplication of a matrix and a vector.
    '''
    def __init__(self, matrix, vector):
        if matrix.shape[1] != vector.shape[0]:
            raise ValueError('Cannot multiply matrix and vector of shapes {}, {}'.format(matrix.shape, vector.shape))
        super().__init__(children=[matrix, vector], shape=(matrix.shape[0],))

    def calc_value(self):
        self.value = np.einsum('...jk,...k->...j', self.children[0].value, self.children[1].value)
    
    def child_gradients(self, gradient):
        return [
            np.einsum('...j,...k->...jk', gradient, self.children[1].value),
            np.einsum('...j,...jk->...k', gradient, self.children[0].value)
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