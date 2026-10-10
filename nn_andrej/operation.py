import math
import itertools
_ids = itertools.count()

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self._id = next(_ids)
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

    def __neg__(self):
        return self*-1

    def __sub__(self, other):
        return self + (-other)

    def exp(self):
        x = self.data
        t = math.exp(x)
        out = Value(t, (self, ), 'exp')
        def _backward():
            self.grad += out.data*out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self*(other**-1)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting integer or floating point numbers"
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += other*(self.data**(other-1))*out.grad

        out._backward = _backward

        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
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

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
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

    def __rmul__(self, other):
        return self * other

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'relu')

        def _backward():
            # ReLU'(x) = 1 if x > 0 else 0 (0 at the origin)
            self.grad += (out.data > 0) * out.grad

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