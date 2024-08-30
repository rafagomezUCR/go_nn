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

type NN struct {
	input_nodes   Matrix
	output_nodes  Matrix
	hidden_nodes  Matrix
	learning_rate float64
}

func (n *NN) query(input_list *Matrix, wih *Matrix, who *Matrix) Matrix {
	input_list.transposeMatrix()
	hidden_inputs := wih.matrixMult(input_list)
	hidden_outputs := sigmoidActivation(&hidden_inputs)
	final_inputs := who.matrixMult(&hidden_outputs)
	final_outputs := sigmoidActivation(&final_inputs)
	return final_outputs
}

func main() {
	in := createMatrix(1, 3)
	on := createMatrix(1, 3)
	hn := createMatrix(1, 3)
	lr := 0.3
	n := NN{
		input_nodes:   in,
		output_nodes:  on,
		hidden_nodes:  hn,
		learning_rate: lr,
	}
	wih := Matrix{
		data: [][]float64{
			{0.1, 0.2, 0.3},
			{0.2, 0.5, 0.4},
			{0.2, 0.6, 0.01},
		},
		rows: 3,
		cols: 3,
	}
	who := Matrix{
		data: [][]float64{
			{0.3, 0.1, 0.3},
			{0.2, 0.3, 0.8},
			{0.1, 0.7, 0.01},
		},
		rows: 3,
		cols: 3,
	}
	input_list := Matrix{
		data: [][]float64{
			{0.1, 0.2, 0.5},
		},
		rows: 1,
		cols: 3,
	}
	res := n.query(&input_list, &wih, &who)
	res.printMatrix()
}

func (a *Matrix) printMatrix() {
	for i := range a.rows {
		for j := range a.cols {
			fmt.Print(a.data[i][j], " ")
		}
		fmt.Println()
	}
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

func feedFoward(m Matrix, weightMatrix *[]Matrix) (Matrix, []Matrix) {
	output_matrix := []Matrix{}
	output_matrix = append(output_matrix, m)
	network_length := len(*weightMatrix)
	for i := range network_length {
		m = (*weightMatrix)[i].matrixMult(&m)
		m.sigmoidActivation()
		output_matrix = append(output_matrix, m)
	}
	return m, output_matrix
}

// func getErrorMatrix(weight []Matrix, err *Matrix) []Matrix {
// 	err_matrix := []Matrix{}
// 	for i := range len(weight) {
// 		sums := []float64{}
// 		for i := range weight[i].rows {
// 			sum := 0.0
// 			for j := range weight[i].cols {
// 				sum += weight[i].data[i][j]
// 			}
// 			sums = append(sums, sum)
// 		}
// 		for i := range weight[i].rows {
// 			for j := range weight[i].cols {
// 				weight[i].data[i][j] /= sums[i]
// 			}
// 		}
// 		weight[i].transposeMatrix()
// 		e := weight[i].matrixMult(err)
// 		err_matrix = append(err_matrix, e)
// 		//return e
// 	}
// 	return err_matrix
// }

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

func (o *Matrix) calculateError(t *Matrix) Matrix {
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
