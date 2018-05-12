import graph
import node
import activation

x = node.Variable(2.0)
y = node.Variable(2.0)

one = node.Constant(1.0)
res = (y+y) + 2.0*( (x+one)*(x+one) )
res = activation.relu(res)

g = graph.Graph(res)

g.visualize()
g.calc_values()
g.calc_gradients()
print('value={}'.format(res.value))
print('x={x}, y={y}: d/dx={dx}, d/dy={dy}'.format(x=x.value, y=y.value, dx=g.gradients[x], dy=g.gradients[y]))
