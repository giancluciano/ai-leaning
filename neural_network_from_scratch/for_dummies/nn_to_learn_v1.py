from numpy import zeros, dot, maximum, max, exp, sum, log, clip, argmax, mean

class Neuron:
    bias = 0
    weight = 0.0

    def forward(self, input):
        print(f"z = {self.weight:.2} * {input} + {self.bias}")
        z = self.weight*input+self.bias
        print(f"a = relu( {z} )")
        return max(z, 0)

class Layer:
    neuron = Neuron()

    def forward(self, input):
        return self.neuron.forward(input)

class NeuralNetworkDummy:
    first = Layer()
    last = Layer()

    def train(self, input, target):
        print(f"Layer 1")
        print(f"a(1)=w(1) * input + b(1)")
        a1 = self.first.forward(input=input)
        print(f"Layer 2")
        print(f"a(2)=w(2) * a(1) + b(2)")
        a2 = self.last.forward(a1)
        print(f"Cost={exp(a2-target):.2}")

if __name__ == "__main__":
    nn = NeuralNetworkDummy()
    nn.train(1,1)