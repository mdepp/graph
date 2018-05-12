import graph
import node

x = node.Variable(2)
y = node.Variable(2)

one = node.Constant(1)
res = (y+y) + 2*( (x+one)*(x+one) + one - one + one - one )

g = graph.Graph(res)

g.visualize()
g.calc_values()
g.calc_gradients()
print('value={}'.format(res.value))
print('x={x}, y={y}: d/dx={dx}, d/dy={dy}'.format(x=x.value, y=y.value, dx=g.gradients[x], dy=g.gradients[y]))
