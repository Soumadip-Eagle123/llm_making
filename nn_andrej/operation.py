import math 
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0,
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"
    '''
        def grad(self, root):
    
            if root is self:
                return 1.0
    
            if root._op == '':
                return 0.0
    
            children = list(root._prev)
    
            var1 = children[0]
            var2 = children[1] if len(children) > 1 else None
    
            if root._op == '*':
                return (
                    self.grad(var1) * var2.data
                    + self.grad(var2) * var1.data
                )
            elif root._op == '+':
                return (
                    self.grad(var1)
                    + self.grad(var2)
                )
            elif root._op == 'tanh':
              return (
                self.grad(var1)
                 *
                (1 - root.data ** 2)
              )
    ''' 
    
    def tanh(self):
      x = self.data
      t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
      out = Value(t, (self, ), 'tanh')

      def _backward():
          self.grad = (1-t**2)*(out.grad)
      out._backward = _backward
      return out

    def __add__(self, other):
        out = Value(
            self.data + other.data,
            (self, other),
            '+'
        )

        def _backward():
            self.grad = 1.0*out.grad
            other.grad = 1.0*out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(
            self.data * other.data,
            (self, other),
            '*'
        )

        def _backward():
            self.grad = out.grad*other.grad
            out.grad = out.grad*self.grad
        out._backward = _backward
        return out