package main

import (
	"fmt"
	"math"
	"math/rand"
)

type Matrix struct {
	data [][]float64
	rows int
	cols int
}

func query(input_list *Matrix, weight_matrices *[]Matrix) Matrix {
	input_list_t := transposeMatrix(input_list)
	for i := range len(*weight_matrices) {
		signal_output := (*weight_matrices)[i].matrixMult(&input_list_t)
		signal_output.sigmoidActivation()
		input_list_t = signal_output
	}
	return input_list_t
}

func train(input_list *Matrix, weight_matrices *[]Matrix, target_list *Matrix) {
	network_output := query(input_list, weight_matrices)
	error_output := calculateError(&network_output, target_list)
	for i := len(*weight_matrices) - 1; i > 0; i-- {
		weight_t := transposeMatrix(&(*weight_matrices)[i])
		error_hidden := weight_t.matrixMult(&error_output)
		// right here goes gradient descent
		error_output = error_hidden
	}
}

func main() {
	//lr := 0.3
	weight_matrices := initializeWeights([]int{2, 3}, 3, 3)
	input_list := Matrix{
		data: [][]float64{
			{0.9, 0.1, 0.8},
		},
		rows: 1,
		cols: 3,
	}
	target_list := Matrix{
		data: [][]float64{
			{1.0},
			{2.0},
			{1.0},
		},
		rows: 3,
		cols: 1,
	}
	train(&input_list, &weight_matrices, &target_list)
}

func (a *Matrix) printMatrix() {
	for i := range a.rows {
		for j := range a.cols {
			fmt.Print(a.data[i][j], " ")
		}
		fmt.Println()
	}
}

func add_matrices(a *Matrix, b *Matrix) Matrix {
	res := createMatrix(a.rows, a.cols)
	for i := range a.rows {
		for j := range a.cols {
			res.data[i][j] = a.data[i][j] + b.data[i][j]
		}
	}
	return res
}

func (a *Matrix) matrixMult(b *Matrix) Matrix {
	if a.cols != b.rows {
		fmt.Println("rows and cols do not match")
		return Matrix{}
	}
	res := createMatrix(a.rows, b.cols)
	for i := range a.rows {
		for j := range b.cols {
			for k := range b.rows {
				res.data[i][j] += a.data[i][k] * b.data[k][j]
			}
		}
	}
	return res
}

func (matrix *Matrix) sigmoidActivation() {
	for i := range matrix.rows {
		for j := range matrix.cols {
			matrix.data[i][j] = 1 / (1 + math.Exp(-matrix.data[i][j]))
		}
	}
}

func sigmoidActivation(matrix *Matrix) Matrix {
	result_matrix := createMatrix(matrix.rows, matrix.cols)
	for i := range matrix.rows {
		for j := range matrix.cols {
			result_matrix.data[i][j] = 1 / (1 + math.Exp(-matrix.data[i][j]))
		}
	}
	return result_matrix
}

func (a *Matrix) scaleMatrix(factor float64) {
	for i := range a.rows {
		for j := range a.cols {
			a.data[i][j] *= factor
		}
	}
}

func (a *Matrix) transposeMatrix() {
	b := createMatrix(a.cols, a.rows)
	for i := range a.rows {
		for j := range a.cols {
			b.data[j][i] = a.data[i][j]
		}
	}
	*a = b
}

func transposeMatrix(a *Matrix) Matrix {
	b := createMatrix(a.cols, a.rows)
	for i := range a.rows {
		for j := range a.cols {
			b.data[j][i] = a.data[i][j]
		}
	}
	return b
}

func initializeWeights(neurons_per_layer []int, input_layer_size int, output_layer_size int) []Matrix {
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
				weightMatrices[i].data[j][k] = rand.Float64()
			}
		}
	}
	return weightMatrices
}

func calculateError(o *Matrix, t *Matrix) Matrix {
	if o.rows != t.rows && o.cols != t.cols {
		fmt.Println("output and target matrices dont match in length")
		return Matrix{}
	}
	e := createMatrix(o.rows, o.cols)
	for i := range o.rows {
		for j := range o.cols {
			e.data[i][j] = math.Pow(t.data[i][j]-o.data[i][j], 2)
		}
	}
	return e
}

func createMatrix(rows int, cols int) Matrix {
	m := Matrix{
		rows: rows,
		cols: cols,
	}
	m.data = make([][]float64, rows)
	for i := range rows {
		m.data[i] = make([]float64, cols)
	}
	return m
}
