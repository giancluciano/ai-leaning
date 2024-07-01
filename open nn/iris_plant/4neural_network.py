''' 
Artificial Intelligence Techniques SL	
artelnics@artelnics.com	

Your model has been exported to this python file.
You can manage it with the 'NeuralNetwork' class.	
Example:

	model = NeuralNetwork()	
	sample = [input_1, input_2, input_3, input_4, ...]	
	outputs = model.calculate_output(sample)


Inputs Names: 	
	0) sepal_lenght
	1) sepal_width
	2) petal_lenght
	3) petal_width


You can predict with a batch of samples using calculate_batch_output method	
IMPORTANT: input batch must be <class 'numpy.ndarray'> type	
Example_1:	
	model = NeuralNetwork()	
	input_batch = np.array([[1, 2], [4, 5]])	
	outputs = model.calculate_batch_output(input_batch)
Example_2:	
	input_batch = pd.DataFrame( {'col1': [1, 2], 'col2': [3, 4]})	
	outputs = model.calculate_batch_output(input_batch.values)
''' 


import math
import numpy as np


class NeuralNetwork:
	def __init__(self):
		self.inputs_number = 4
		self.inputs_names = ['sepal_lenght', 'sepal_width', 'petal_lenght', 'petal_width']


	def calculate_outputs(self, inputs):
		sepal_lenght = inputs[0]
		sepal_width = inputs[1]
		petal_lenght = inputs[2]
		petal_width = inputs[3]

		scaled_sepal_lenght = (sepal_lenght-5.843333244)/0.8280661106
		scaled_sepal_width = (sepal_width-3.057333231)/0.4358662963
		scaled_petal_lenght = (petal_lenght-3.757999897)/1.765298247
		scaled_petal_width = (petal_width-1.19933331)/0.762237668
		
		perceptron_layer_1_output_0 = np.tanh( -0.126343 + (scaled_sepal_lenght*0.452364) + (scaled_sepal_width*-1.75141) + (scaled_petal_lenght*1.83151) + (scaled_petal_width*1.67542) )
		perceptron_layer_1_output_1 = np.tanh( -1.83326 + (scaled_sepal_lenght*-2.34547) + (scaled_sepal_width*3.95177) + (scaled_petal_lenght*-3.21016) + (scaled_petal_width*-2.40714) )
		perceptron_layer_1_output_2 = np.tanh( 3.03453 + (scaled_sepal_lenght*0.751756) + (scaled_sepal_width*0.960947) + (scaled_petal_lenght*-3.69685) + (scaled_petal_width*-1.57027) )
		
		probabilistic_layer_combinations_0 = -0.830248 -6.19397*perceptron_layer_1_output_0 +0.704016*perceptron_layer_1_output_1 +3.62503*perceptron_layer_1_output_2 
		probabilistic_layer_combinations_1 = 1.81044 +1.21444*perceptron_layer_1_output_0 -1.41582*perceptron_layer_1_output_1 +1.55054*perceptron_layer_1_output_2 
		probabilistic_layer_combinations_2 = -1.24549 +5.26503*perceptron_layer_1_output_0 +0.714832*perceptron_layer_1_output_1 -4.87547*perceptron_layer_1_output_2 
			
		sum = np.exp(probabilistic_layer_combinations_0) + np.exp(probabilistic_layer_combinations_1) + np.exp(probabilistic_layer_combinations_2)
		
		iris_setosa = np.exp(probabilistic_layer_combinations_0)/sum
		iris_versicolor = np.exp(probabilistic_layer_combinations_1)/sum
		iris_virginica = np.exp(probabilistic_layer_combinations_2)/sum
		
		out = [None]*3

		out[0] = iris_setosa
		out[1] = iris_versicolor
		out[2] = iris_virginica

		return out


	def calculate_batch_output(self, input_batch):
		output_batch = [None]*input_batch.shape[0]

		for i in range(input_batch.shape[0]):

			inputs = list(input_batch[i])

			output = self.calculate_outputs(inputs)

			output_batch[i] = output

		return output_batch
