from graphviz import Digraph
from operation import Value


def trace(root):
    """
    Build a set of all nodes and edges in the computational graph.
    """
    nodes = set()
    edges = set()

    def build(v):
        if v not in nodes:
            nodes.add(v)

            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)

    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='jpg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)

    # Create a node for every Value
    for n in nodes:
        uid = str(id(n))

        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), 
            shape='record'
        )

        # If this Value was created by an operation,
        # create a separate operation node
        if n._op:
            dot.node(
                name=uid + n._op,
                label=n._op
            )

            # Operation -> Value
            dot.edge(
                uid + n._op,
                uid
            )

    # Value -> Operation
    for n1, n2 in edges:
        dot.edge(
            str(id(n1)),
            str(id(n2)) + n2._op
        )

    return dot

if __name__ == "__main__":
   
   a = Value(2.0, label='a')
   b = Value(3.0, label='b')
   c = Value(1.0, label='c')
   
   p = a*b
   p.label = 'p'
   print(p)
   print(p._prev)
   q = p + c
   q.label = 'q'
   print(q)
   print(q._prev)
   
   d = Value(6.0, label='d')
   L = d*q
   L.label = 'L'
   
   filename = draw_dot(L).render('op_graph', view=True)
   print("Generated file: "+filename)
   
   