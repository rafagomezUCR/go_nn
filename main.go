package main

import (
	"fmt"
	"math"
)

type matrix [][]float64

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
	res := make(matrix, a_rows)
	for i := range res {
		res[i] = make([]float64, b_cols)
	}
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
	output.printMatrix()
	for i := range len(output) {
		for j := range len(output[i]) {
			output[i][j] = sigmoidActivation(output[i][j])
		}
	}
	return output
}

func main() {
	fmt.Println("printing")
	a := matrix{
		{1, 2, 3},
		{4, 5, 6},
	}
	b := matrix{
		{7, 8},
		{9, 10},
		{11, 12},
	}
	c := a.matrixMult(&b)
	c.printMatrix()
	i := matrix{
		{0.9, 0.3, 0.4},
		{0.2, 0.8, 0.2},
		{0.1, 0.5, 0.6},
	}
	w := matrix{
		{0.9},
		{0.1},
		{0.8},
	}
	o := feedFoward(&i, &w)
	o.printMatrix()
}
