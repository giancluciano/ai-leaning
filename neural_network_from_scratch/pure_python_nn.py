# example of what happen inside a single neuron
inputs = [1,2,3]
weights = [0.2, 0.8, -0.5]
bias = 2

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias 


# example of what happen inside a single layer with 3 neurons
inputs = [1,2,3]
weights1 = [0.2, 0.8, -0.5]
weights2 = [0.5, -0.91, 0.26]
weights3 = [-0.26, -0.27, 0.87]
weights = [weights1, weights2, weights3]
bias1 = 2
bias2 = 3
bias3 = 0.5
bias = [bias1, bias2, bias3]

output1 = inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + bias1
output2 = inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + bias2
output3 = inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + bias3
output = [output1, output2, output3]

# example of what happen inside a single layer with 3 neurons and 3 input batches
input1 = [1,2,3,2.5]
input2 = [2,5,-1,2]
input3 = [-1.5,2.7,3.3,-0.8]
inputs = [input1, input2, input3]
weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
weights = [weights1, weights2, weights3]
bias1 = 2
bias2 = 3
bias3 = 0.5
bias = [bias1, bias2, bias3]

output = []

for i in inputs:
    _len = len(i)
    n_output = []
    for index, w in enumerate(weights):
        n_output.append(sum([i[o]*w[o] for o in range(_len)]) + bias[index])
    output.append(n_output)

print(f"output: {output}")
