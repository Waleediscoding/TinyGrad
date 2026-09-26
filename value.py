import math

class Value:
    def __init__(self, data, _children=()):
        self.data = data
        self._prev = _children
        self.grad = 0
        self._backward = lambda: None
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(self.data + other.data, (self, other))
        
        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(self.data * other.data, (self, other))
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * (-1)
    
    def __sub__(self, other):
        return other + (-self)
    
    def __rsub__(self, other):
        return self - other
    
    def __pow__(self, power):
        assert isinstance(power, int) or isinstance(power, float), "Currently only supporting ints and floats."
        
        out = Value(self.data ** power, (self,))
        
        def _backward():
            self.grad = (power * (self.data ** (power - 1))) * out.grad
            
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def tanh(self):
        value = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1) # tanh(self.data)
        
        out = Value(value, (self,))
        
        def backward():
            self.grad += (1 - (out.data ** 2)) * out.grad
            
        out._backward = backward
        return out
    
    def backward(self):
        self.grad = 1
        
        topo = []
        visited = set()
        
        def toposort(node):
            if node not in visited:
                visited.add(node)
                
                for child in node._prev:
                    toposort(child)
                topo.append(node)

        toposort(self)
        for node in reversed(topo):
            node._backward()