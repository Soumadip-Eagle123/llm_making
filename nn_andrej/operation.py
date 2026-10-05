import math


class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None


    def __repr__(self):
        return f"Value(data={self.data})"
    def tanh(self):

        x = self.data

        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

        out = Value(
            t,
            (self,),
            'tanh'
        )

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward

        return out

    def __add__(self, other):

        out = Value(
            self.data + other.data,
            (self, other),
            '+'
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):

        out = Value(
            self.data * other.data,
            (self, other),
            '*'
        )

        def _backward():

            # d(self * other)/dself = other
            # d(self * other)/dother = self

            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out


    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):

            if v not in visited:

                visited.add(v)

                # First visit all dependencies
                for child in v._prev:
                    build_topo(child)

                # Then add current node
                topo.append(v)

        # Start from the final output
        build_topo(self)

        # dL/dL = 1
        self.grad = 1.0

        # Traverse graph backwards
        for node in reversed(topo):

            node._backward()