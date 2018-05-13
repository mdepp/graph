import graph
import node
import activation
import numpy as np

length = 5
width = 4

# x = node.Variable(np.random.random((length,)))
# y = node.Variable(np.random.random((width,)))

# A = node.Variable(np.random.random((width, length)))

# one = node.Constant(np.ones((length,)))

# res = (y+y) * 2..0.*( A*((x+one)*(x+one)) )
# res = activation.relu(res)

W = node.Variable(np.array([ [[2., 1.],[1., 2.]] ]))
b = node.Variable(np.array([ [0., 0.] ]))
x = node.Variable(np.array([ [2.,1.], [1., 2.] ]))
product = (W*x)
res = product*product + b

g = graph.Graph(res)

g.visualize()
g.calc_values()
g.calc_gradients()
print('value={}'.format(res.value))
# print('x={x}, y={y}: d/dx={dx}, d/dy={dy}'.format(x=x.value, y=y.value, dx=g.gradients[x], dy=g.gradients[y]))
print('d/dW = \n{}'.format(g.gradients[W]))
print('d/db = \n{}'.format(g.gradients[b]))
print('d/dx = \n{}'.format(g.gradients[x]))


# TODO: A few bugs..
#   - Matrix multiplication is a bit broken, since rmul assumes commutativity
#   - Node ctor says it converts non-arrays to arrays, but does not
#   - __add__ needs a doc comment
#   - rsub is not implemented
#   - Variable still has substitute()
#   - No shape tests in ScalarMultiply, MatVecMultiply
#   - Functor is still a bit of an ugly hack and should be moved