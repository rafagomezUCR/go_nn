package main

import "math/rand/v2"

type NN struct {
	weights       []Matrix
	activation    string
	input_layer   Matrix
	target        Matrix
	hidden_layers []int
	learning_rate float64
	epochs        int
}

func (nn *NN) feed_foward() []Matrix {
	output_list := make([]Matrix, 0)
	input_list_t := transposeMatrix(&nn.input_layer)
	output_list = append(output_list, input_list_t)
	for i := range len(nn.weights) {
		signal_output := nn.weights[i].matrixMult(&input_list_t)
		signal_output.sigmoidActivation()
		output_list = append(output_list, signal_output)
		input_list_t = signal_output
	}
	return output_list
}

func (nn *NN) train() {
	network_outputs := nn.feed_foward()
	error_output := calculateError(&network_outputs[len(network_outputs)-1], &nn.target)
	for i := len(nn.weights) - 1; i >= 0; i-- {
		weight_t := transposeMatrix(&(nn.weights)[i])
		error_hidden := weight_t.matrixMult(&error_output)
		// right here goes gradient descent
		one_subtract_o := scalar_minus_matrix(1.0, &network_outputs[i+1])
		de_dw := elementwise_matrix_multiplication(&network_outputs[i+1], &one_subtract_o)
		de_dw = elementwise_matrix_multiplication(&error_output, &de_dw)
		hidden_outputs := transposeMatrix(&network_outputs[i])
		de_dw = de_dw.matrixMult(&hidden_outputs)
		de_dw = scale_matrix(nn.learning_rate, &de_dw)
		nn.weights[i] = add_matrices(&(nn.weights)[i], &de_dw)
		error_output = error_hidden
	}
}

func (nn *NN) query() Matrix {
	input_list_t := transposeMatrix(&nn.input_layer)
	for i := range len(nn.weights) {
		signal_output := nn.weights[i].matrixMult(&input_list_t)
		signal_output.sigmoidActivation()
		input_list_t = signal_output
	}
	return input_list_t
}

func initializeWeights(input_layer_size int, neurons_per_layer []int, output_layer_size int) []Matrix {
	weightMatrices := make([]Matrix, 0)
	for i := range len(neurons_per_layer) + 1 {
		if i == 0 {
			w := createMatrix(neurons_per_layer[i], input_layer_size)
			weightMatrices = append(weightMatrices, w)
		} else if i == len(neurons_per_layer) {
			w := createMatrix(output_layer_size, neurons_per_layer[i-1])
			weightMatrices = append(weightMatrices, w)
		} else {
			w := createMatrix(neurons_per_layer[i], neurons_per_layer[i-1])
			weightMatrices = append(weightMatrices, w)
		}
	}
	for i := range len(weightMatrices) {
		for j := range weightMatrices[i].rows {
			for k := range weightMatrices[i].cols {
				weightMatrices[i].data[j][k] = rand.Float64() - 0.5
			}
		}
	}
	return weightMatrices
}
