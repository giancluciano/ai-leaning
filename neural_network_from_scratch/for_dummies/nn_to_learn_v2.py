from numpy import zeros, dot, maximum, max, exp, sum, log, clip, argmax, mean

class Neuron:
    bias = 0
    weight = 0.5

    def forward(self, input):
        mult = self.weight*input
        z = mult+self.bias
        relu = max(z, 0)
        print(f"z = {self.weight:.2} * {input} + {self.bias}")
        print(f"a = relu( {z} ) -> 1(z > 0)")
        dvalue = 1.0 # number from next layer
        # Derivative of ReLU and the chain rule
        drelu_dz = dvalue * (1. if z > 0 else 0.)
        # Partial derivatives of the multiplication, the chain rule
        dsum_dxw = 1
        dsum_db = 1
        drelu_dxw = drelu_dz * dsum_dxw
        drelu_db = drelu_dz * dsum_db

        dmul_dx = self.weight
        drelu_dx = drelu_dxw * dmul_dx
        dmul_dw = input
        drelu_dx = drelu_dxw * dmul_dx
        drelu_dw = drelu_dxw * dmul_dw
        self.weight += 0.1 * drelu_dw
        #print(drelu_dx)
        return relu

class Layer:
    neuron = Neuron()

    def forward(self, input):
        return self.neuron.forward(input)

class NeuralNetworkDummy:
    layer = Layer()

    def train(self, input, target):
        print(f"Layer 1")
        print(f"a(1)=w(1) * input + b(1)")
        a1 = self.layer.forward(input=input)
        #import ipdb; ipdb.set_trace()
        print(f"Cost={exp(a1-target):.2}")
        a2 = self.layer.forward(input=input)
        print(f"Cost={exp(a2-target):.2}")

if __name__ == "__main__":
    nn = NeuralNetworkDummy()
    nn.train(1,1)