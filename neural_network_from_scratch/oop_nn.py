
# using OOO
from numpy.random import seed
from nnfs.datasets import vertical_data
seed(0)

from models import NeuralNetwork

train_data = vertical_data(1000, 2)
test_data = vertical_data(10, 2)


nn = NeuralNetwork(layers_shape = (2,64,64,2))
nn.train(train_data[0], train_data[1])
print(nn.predict(test_data[0]))
print(test_data[1])
#nn.calculate_loss()