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

# res = (y+y) * 2.0*( A*((x+one)*(x+one)) )
# res = activation.relu(res)

A = node.Constant(np.array([[0.9, -1],[1, 0.9]]))
x = node.Variable(np.array([0.3,0.4]))
res = A*x

g = graph.Graph(res)

g.visualize()
g.calc_values()
g.calc_gradients()
print('value={}'.format(res.value))
# print('x={x}, y={y}: d/dx={dx}, d/dy={dy}'.format(x=x.value, y=y.value, dx=g.gradients[x], dy=g.gradients[y]))
print('d/dA = {}'.format(g.gradients[A]))
print('d/dx = {}'.format(g.gradients[x]))