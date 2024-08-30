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
	input := Matrix{
		data: [][]float64{
			{0.9, 0.2, 0.3},
		},
		rows: 1,
		cols: 3,
	}
	input.printMatrix()
	target := Matrix{
		data: [][]float64{
			{1, 0.7, 0.9},
		},
		rows: 1,
		cols: 3,
	}
	target.transposeMatrix()
	input.transposeMatrix()
	input.printMatrix()
	wm := initializeWeights(2)
	for i := range len(wm) {
		wm[i].printMatrix()
		fmt.Println()
	}
	for i := range 2 {
		input = feedFoward(&input, &wm[i])
	}
	fmt.Println()
	e := input.calculateError(&target)
	e.printMatrix()
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
	//output.printMatrix()
	return output
}

func errorBackpropagation(weight *Matrix, error *Matrix) Matrix {

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
