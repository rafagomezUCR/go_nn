package main

import (
	"fmt"
	"math"
	"math/rand"
)

type matrix [][]float64

func main() {
	input := matrix{
		{0.9, 0.2, 0.3},
	}
	target := matrix{
		{1, 0.7, 0.9},
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
		input = feedFoward(&wm[i], &input)
	}
	fmt.Println()
	e := input.calculateError(&target)
	e.printMatrix()
}

func (a *matrix) printMatrix() {
	for i := range len(*a) {
		for j := range len((*a)[0]) {
			fmt.Print((*a)[i][j], " ")
		}
		fmt.Println()
	}
}

func (a *matrix) matrixMult(b *matrix) matrix {
	a_rows, a_cols := len(*a), len((*a)[0])
	b_rows, b_cols := len(*b), len((*b)[0])
	if a_cols != b_rows {
		fmt.Println("rows and cols do not match")
		return matrix{}
	}
	res := createMatrix(a_rows, b_cols)
	for i := range a_rows {
		for j := range b_cols {
			for k := range b_rows {
				res[i][j] += (*a)[i][k] * (*b)[k][j]
			}
		}
	}
	return res
}

func sigmoidActivation(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func feedFoward(a *matrix, weight *matrix) matrix {
	output := a.matrixMult(weight)
	for i := range len(output) {
		for j := range len(output[i]) {
			output[i][j] = sigmoidActivation(output[i][j])
		}
	}
	output.printMatrix()
	return output
}

func (a *matrix) transposeMatrix() {
	a_rows, a_cols := len(*a), len((*a)[0])
	b := createMatrix(a_cols, a_rows)
	for i := range a_rows {
		for j := range a_cols {
			b[j][i] = (*a)[i][j]
		}
	}
	*a = b
}

// right now all hidden layers will have 3 nodes
// later im going to allow to customize the number of nodes per layer
func initializeWeights(hiddenLayers int) []matrix {
	weightMatrices := make([]matrix, 0)
	for range hiddenLayers {
		w := createMatrix(3, 3)
		weightMatrices = append(weightMatrices, w)
	}
	for i := range len(weightMatrices) {
		for j := range len(weightMatrices[i]) {
			for k := range len(weightMatrices[i][j]) {
				weightMatrices[i][j][k] = rand.Float64()
			}
		}
	}
	return weightMatrices
}

func (o *matrix) calculateError(t *matrix) matrix {
	o_rows, o_cols := len(*o), len((*o)[0])
	t_rows, t_cols := len(*t), len((*t)[0])
	if o_rows != t_rows && o_cols != t_cols {
		fmt.Println("output and target matrices dont match in length")
		return matrix{}
	}
	e := createMatrix(o_rows, o_cols)
	t.printMatrix()
	o.printMatrix()
	for i := range o_rows {
		for j := range o_cols {
			e[i][j] = math.Pow((*t)[i][j]-(*o)[i][j], 2)
		}
	}
	return e
}

func createMatrix(rows int, cols int) matrix {
	m := make(matrix, rows)
	for i := range len(m) {
		m[i] = make([]float64, cols)
	}
	return m
}
