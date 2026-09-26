import random
from value import Value

class Neuron:
    def __init__(self, n):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        z = sum((xi * wi for xi, wi in zip(x, self.w)), self.b) # Pre-activation
        out = z.tanh()
        
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n, ns):
        self.l = [Neuron(n) for _ in range(ns)]
        
    def __call__(self, x):
        out = [n(x) for n in self.l]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.l for p in n.parameters()]

class MLP:
    def __init__(self, mlp): # mlp: [input, layer1...n, output]
        self.ls = [Layer(mlp[i], mlp[i+1]) for i in range(len(mlp) - 1)]
        
    def __call__(self, x):
        for l in self.ls:
            x = l(x)
        
        return x
    
    def parameters(self):
        return [p for l in self.ls for p in l.parameters()]