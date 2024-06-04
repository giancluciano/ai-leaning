from numpy import zeros, dot, maximum, max, exp, sum, log, clip, argmax, mean, multiply, array
from numpy.random import randn
from math import e
from typing import List
from copy import deepcopy

class NeuralNetworkBase:

    def forward(self, layer, input):
        # simple linear activation
        return dot(input, layer[0]) + layer[1]

    def forward_relu(self, layer, input):
        # Rectifier activation
        return maximum(0, self.forward(layer, input))

    def forward_softmax(self, layer, input:List[List[int]]):
        # Softmax activation
        # normalized distribution of probabilities
        output = self.forward(layer, input)
        exp_output = exp(output - max(output, axis=1, keepdims=True))
        return  exp_output / sum(exp_output, axis=1, keepdims=True) #axis=1 if input is a batch

    def get_random_layer(self, n_inputs, layer_neurons):
        # create random weights and biases  
        return (0.1 * randn(n_inputs, layer_neurons)), zeros((1, layer_neurons))

    def shuffle_neurons(self, layer):
        # update weights in a random directions  
        layer[0] += 0.05 * randn(len(layer[0]), len(layer[0][0]))
        layer[1] += 0.05 * randn(1, len(layer[0][0]))

    def backpropagate(self, layer, input, output, dcost_a):
        #Derivatives list
        # dcost_a from next layer OR 2(a-y)
        da_z = [ [1.] if o[0] > 0 else [0.] for o in output]
        dz_weight = input
        dcost_weight = dz_weight * da_z * dcost_a
        dcost_bias = 1 * da_z * dcost_a
        import ipdb; ipdb.set_trace()
        dcost_input = layer[0] * da_z * dcost_a
        #dcost_input =  dot(dcost_a, layer[0])
        layer[0] += -0.001 * dcost_weight
        layer[1] += -0.001 * dcost_bias
        print(f"z = {layer[0]:.2} * {input} + {layer[1]}")
        return dcost_input
    

class NeuralNetwork(NeuralNetworkBase):

    def __init__(self, layers_shape) -> None:
        self.layers_shape = layers_shape
        self.layers = []
    
    def predict(self, inputs):
        output = inputs
        for layer in self.layers:
            output = self.forward_softmax(layer, output)
        return argmax(output, axis=1)

    def relu_train(self, inputs, targets, interations=10000):
        n_inputs = len(inputs[0])
        for layer_neurons in self.layers_shape:
            self.layers.append(self.get_random_layer(n_inputs, layer_neurons))
            n_inputs = layer_neurons
        
        accuracy = 0
        for _ in range(interations):
            outputs = [inputs]
            for layer in self.layers:
                outputs.append(self.forward_relu(layer, outputs[-1]))

            output = outputs.pop()
            prediction = argmax(output, axis=1)
            cost = (prediction - targets)**2
            dcost = array([multiply(2, prediction-targets)]).T
            for layer in self.layers[::-1]:
                import ipdb; ipdb.set_trace()
                self.backpropagate(layer, outputs.pop(), output, dcost)

        print(f"best accuracy {accuracy}")


    def softmax_train(self, inputs, targets, interations=10000):
        n_inputs = len(inputs[0])
        for layer_neurons in self.layers_shape:
            self.layers.append(self.get_random_layer(n_inputs, layer_neurons))
            n_inputs = layer_neurons
        
        loss = 99
        best_output = None
        for _ in range(interations):
            output = inputs
            current_layers = deepcopy(self.layers)
            for layer in current_layers:
                self.shuffle_neurons(layer)
                output = self.forward_softmax(layer, output)

            current_loss = mean(-log(clip(output[range(len(output)), targets], 1e-7, 1-1e-7)))
            if current_loss < loss:
                
                loss = current_loss
                self.layers = deepcopy(current_layers)
                best_output = deepcopy(output)
        print(f"best loss {loss}") # Categorical cross-entropy
        print(f"best accuracy {mean(argmax(best_output, axis=1) == targets)}")