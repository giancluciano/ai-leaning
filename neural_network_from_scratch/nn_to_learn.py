from numpy import max, multiply


class Neuron:
    bias = 0
    weight = 0.5

    def forward(self, input):
        mult = self.weight*input
        z = mult+self.bias
        relu = max(z, 0)
        print(f"z = {self.weight:.2} * {input} + {self.bias}")
        print(f"a = relu( {z} ) -> 1(z > 0)")
        
        return relu
    
    def backpropagate(self, input, output, dvalue):
        #Derivatives list
        # dvalue from next layer OR 2(a-y)
        dcost_a = dvalue
        da_z = (1. if output > 0 else 0.)
        dz_weight = input
        dcost_weight = dz_weight * da_z * dcost_a
        dcost_bias = 1 * da_z * dcost_a
        dcost_input = self.weight * da_z * dcost_a
        self.weight += -0.001 * dcost_weight
        #self.bias += -0.001 * dcost_bias
        print(f"z = {self.weight:.2} * {input} + {self.bias}")

class Layer:
    neuron = Neuron()

    def forward(self, input):
        return self.neuron.forward(input)

    def backpropagate(self, input, output, dvalue):
        self.neuron.backpropagate(input,output, dvalue)

class NeuralNetworkDummy:
    layer = Layer()

    def train(self, input, target):
        cost = 99
        while cost != 0:
            print(f"Layer 1")
            print(f"a(1)=w(1) * input + b(1)")
            a1 = self.layer.forward(input=input)

            cost = (a1-target)**2
            dcost = multiply(2, (a1-target))
            print(f"Cost={cost:.2}")
            print(f"dCost={dcost:.2}")
            self.layer.backpropagate(input, a1, dcost)


if __name__ == "__main__":
    nn = NeuralNetworkDummy()
    nn.train(1,1)