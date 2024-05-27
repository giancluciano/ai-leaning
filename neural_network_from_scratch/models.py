from numpy import zeros, dot, maximum, max, exp, sum, log, clip, argmax, mean
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
        return [(0.1 * randn(n_inputs, layer_neurons)), zeros((1, layer_neurons))]

    def shuffle_neurons(self, layer):
        # update weights in a random directions  
        layer[0] += 0.05 * randn(len(layer[0]), len(layer[0][0]))
        layer[1] += 0.05 * randn(1, len(layer[0][0]))
    

class NeuralNetwork(NeuralNetworkBase):

    def __init__(self, layers_shape) -> None:
        self.layers_shape = layers_shape
        self.layers = []
    
    def predict(self, inputs):
        output = inputs
        for layer in self.layers:
            output = self.forward_softmax(layer, output)
        return argmax(output, axis=1)

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
            n_inputs = len(inputs[0])
            for layer in current_layers:
                self.shuffle_neurons(layer)
                output = self.forward_softmax(layer, output)
                n_inputs = layer_neurons

            current_loss = mean(-log(clip(output[range(len(output)), targets], 1e-7, 1-1e-7)))
            if current_loss < loss:
                
                loss = current_loss
                self.layers = deepcopy(current_layers)
                best_output = deepcopy(output)
        print(f"best loss {loss}") # Categorical cross-entropy
        print(f"best accuracy {mean(argmax(best_output, axis=1) == targets)}")