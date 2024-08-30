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

func main() {
	weightMatrixIH := Matrix{
		data: [][]float64{
			{3.0, 2.0},
			{1.0, 7.0},
		},
		rows: 2,
		cols: 2,
	}
	weightMatrixHO := Matrix{
		data: [][]float64{
			{2.0, 3.0},
			{1.0, 4.0},
		},
		rows: 2,
		cols: 2,
	}
	weightMatrix := []Matrix{}
	weightMatrix = append(weightMatrix, weightMatrixIH)
	weightMatrix = append(weightMatrix, weightMatrixHO)
	err_output := Matrix{
		data: [][]float64{
			{0.8},
			{0.5},
		},
		rows: 2,
		cols: 1,
	}
	err_matrix := []Matrix{}
	err_matrix = append(err_matrix, err_output)
	for i := len(weightMatrix) - 1; i >= 0; i-- {
		new_err := getErrorMatrix(&weightMatrix[i], &err_output)
		err_matrix = append(err_matrix, new_err)
		err_output = new_err
	}

	for i := range len(err_matrix) {
		err_matrix[i].printMatrix()
	}

	// input := Matrix{
	// 	data: [][]float64{
	// 		{0.9, 0.2, 0.3},
	// 	},
	// 	rows: 1,
	// 	cols: 3,
	// }
	// input.printMatrix()
	// target := Matrix{
	// 	data: [][]float64{
	// 		{1, 0.7, 0.9},
	// 	},
	// 	rows: 1,
	// 	cols: 3,
	// }
	// target.transposeMatrix()
	// input.transposeMatrix()
	// input.printMatrix()
	// wm := initializeWeights(2)
	// for i := range len(wm) {
	// 	wm[i].printMatrix()
	// 	fmt.Println()
	// }
	// for i := range 2 {
	// 	input = feedFoward(&input, &wm[i])
	// }
	// fmt.Println()
	// e := []Matrix{}
	// e = append(e, input.calculateError(&target))
	// e[0].printMatrix()
	// for i := 1; i < 2; i++ {
	// 	e = append(e, getErrorMatrix(&wm[i], &e[i-1]))
	// }
	// fmt.Println(len(e))
	// for i := range len(e) {
	// 	e[i].printMatrix()
	// }
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

func sigmoidActivation(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func feedFoward(a *Matrix, w *Matrix) Matrix {
	output := w.matrixMult(a)
	for i := range output.rows {
		for j := range output.cols {
			output.data[i][j] = sigmoidActivation(output.data[i][j])
		}
	}
	return output
}

func getErrorMatrix(weight *Matrix, err *Matrix) Matrix {
	sums := []float64{}
	for i := range weight.rows {
		sum := 0.0
		for j := range weight.cols {
			sum += weight.data[i][j]
		}
		sums = append(sums, sum)
	}
	for i := range weight.rows {
		for j := range weight.cols {
			weight.data[i][j] /= sums[i]
		}
	}
	weight.transposeMatrix()
	e := weight.matrixMult(err)
	return e
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

// right now all hidden layers will have 3 nodes
// later im going to allow to customize the number of nodes per layer
func initializeWeights(hiddenLayers int) []Matrix {
	weightMatrices := make([]Matrix, 0)
	for range hiddenLayers {
		w := createMatrix(3, 3)
		weightMatrices = append(weightMatrices, w)
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
