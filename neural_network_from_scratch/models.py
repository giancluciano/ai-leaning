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
        return maximum(0, dot(input, layer[0]) + layer[1])

    def forward_softmax(self, layer, input:List[List[int]]):
        # Softmax activation
        output = dot(input, layer[0]) + layer[1]
        exp_output = exp(output - max(output, axis=1, keepdims=True))
        return  exp_output / sum(exp_output, axis=1, keepdims=True) #axis=1 if input is a batch

    def shuffle_neurons(self, layer):
        layer[0] += 0.05 * randn(len(layer[0]), len(layer[0][0]))
        layer[1] += 0.05 * randn(1, len(layer[0][0]))
    
    def get_random_layer(self, n_inputs, layer_neurons):
            return [(0.1 * randn(n_inputs, layer_neurons)), zeros((1, layer_neurons))]

class NeuralNetwork(NeuralNetworkBase):

    def __init__(self, layers_shape) -> None:
        self.layers_shape = layers_shape
        self.layers = []
        self.best_weights_n_biases = []
    
    def predict(self, inputs):
        output = inputs
        for layer in self.layers:
            output = self.forward_softmax(layer, output)
        return argmax(output, axis=1)

    def train(self, inputs, targets, interations=10000):
        output = inputs
        n_inputs = len(inputs[0])
        for layer_neurons in self.layers_shape:
            self.layers.append(self.get_random_layer(n_inputs, layer_neurons))
            output = self.forward_softmax(self.layers[-1], output)
            n_inputs = layer_neurons
        
        loss = mean(-log(clip(output[range(len(output)), targets], 1e-7, 1-1e-7)))
        
        for _ in range(interations):
            output = inputs
            current_layers = deepcopy(self.layers)
            n_inputs = len(inputs[0])
            for layer in current_layers:
                self.small_shuffle_neurons(layer)
                output = self.forward_softmax(layer, output)
                n_inputs = layer_neurons

            current_loss = mean(-log(clip(output[range(len(output)), targets], 1e-7, 1-1e-7)))
            if current_loss < loss:
                print(f"new best loss {loss}")
                loss = current_loss
                self.layers = deepcopy(current_layers)
