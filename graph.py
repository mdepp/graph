import numpy.matlib as np

class Graph:
    '''
    A Graph is used to calculate values and gradients of nodes. It is initialized
    with a single node (the root). The gradient at each node is defined to be
    the gradient of this root node (with elements summed) with respect to each
    node.
    '''
    def __init__(self, node):
        '''
        Initialize the graph with the root node, and compile it (i.e. create a 
        topological ordering of all child nodes).
        '''
        self.root = node
        self.gradients = [] # Dictionary mapping node -> node gradient
        self.values = []
        self.ordering = self._create_topological_ordering(self.root)

    def compile(self):
        '''
        Re-compile the graph (for example, if structure changes).
        '''
        self.ordering = self._create_topological_ordering(self.root)
    
    def calc_values(self):
        '''
        Calculate the values of the root node and all its children. The values
        are stored in the 'value' attribute of each node.

        TODO: Caching
        '''
        # Loop in reverse topological order (from leaves -> root)
        for node in reversed(self.ordering):
            node.calc_value()

    def calc_gradients(self):
        '''
        Calculate all gradients, and store them in the Graph's 'gradients'
        attribute as a dictionary from node to gradient.
        '''
        self.calc_values()
        self.gradients = {self.root: 1.0}
        # Loop in topological order (from root -> leaves)
        for node in self.ordering:
            child_gradients = node.child_gradients(self.gradients[node])
            for child, gradient in zip(node.children, child_gradients):
                if child not in self.gradients:
                    self.gradients[child] = np.zeros_like(child.value)
                
                self.gradients[child] += gradient


    def visualize(self, filename='graph.gv', view=True):
        '''
        Create and display a visual representation of the graph.
        '''
        from graphviz import Digraph
        dot = Digraph(comment='Computation Graph')

        names = {n: '{}; {}'.format(type(n), i) for i,n in enumerate(self.ordering)}

        for index, node in enumerate(self.ordering):
            dot.node(names[node])
            for child in node.children:
                dot.edge(names[node], names[child])
        
        dot.render(filename, view=view)

    def _create_topological_ordering(self, root):
        '''
        Create and return a topological ordering (as a list of nodes, from root
        to leaves) of root node and all children.
        '''
        # This is an implementation of Kahn's algorithm. Although the graph
        # can have multiple edges between the same nodes, this function only
        # considers the unique edges, which slows running time somewhat, but
        # doesn't change dependencies.

        ordering = []
        S = [root]

        removed_edges = {}

        while S:
            node = S.pop()
            ordering.append(node)

            for child in set(node.children): # Consider all unique children
                if child not in removed_edges:
                    removed_edges[child] = set()
                
                removed_edges[child].add(node)
                if len(removed_edges[child]) == len(set(child.parents)):
                    S.append(child)
        
        return ordering
                

