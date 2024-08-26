
# using OOO
from numpy.random import seed
from nnfs.datasets import vertical_data, spiral_data
from matplotlib import pyplot
import cProfile
import pstats
from nn_to_learn import NeuralNetworkDummy
seed(0)

from models import NeuralNetwork

BATCH_LEN = 1000
INTERATIONS = 10000
N_CLASSES = 2

train_data, target = vertical_data(BATCH_LEN, N_CLASSES)

#pyplot.axis('equal')
#pyplot.scatter(train_data[:,0], train_data[:,1], c=y, cmap='brg')
#pyplot.show()
test_data = vertical_data(10, 2)
# 
# print("train: ")
# seed(0)
# with cProfile.Profile() as pr:
#     nn = NeuralNetwork(layers_shape = (2,1,1, N_CLASSES)) # layers shape is a tuple of int (number of neurons for each layer)
#     nn.train(train_data, target, INTERATIONS)
# 
#     print(f"{pstats.Stats(pr).total_tt:.4} seconds, {pstats.Stats(pr).total_calls} function calls")
#     # best loss 0.11168550211891784
#     # best accuracy 0.955
#     # 6.963818391000001 seconds, 2412600 function calls
# 
print("softmax: ")
seed(0)
with cProfile.Profile() as pr:
    nn = NeuralNetwork(layers_shape = (2,4,4, N_CLASSES)) # layers shape is a tuple of int (number of neurons for each layer)
    nn.softmax_train(train_data, target, INTERATIONS)

    print(f"{pstats.Stats(pr).total_tt:.4} seconds, {pstats.Stats(pr).total_calls} function calls")
    # best loss 0.11168550211891784
    # best accuracy 0.955
    # 6.963818391000001 seconds, 2412600 function calls
